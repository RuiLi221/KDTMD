import numpy as np 
import tensorflow as tf
from tfkan import layers
from tfkan.layers import DenseKAN, Conv1DKAN
from keras_flops import get_flops
import pywt
import matplotlib.pyplot as plt

def disentangle(x, w, j):
    x = x.transpose(0,3,2,1) # [S,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)
    return xl, xh

wave = 'sym2'
level = 1
train_XL, train_XH = disentangle(train_x, wave, level)

W=500
lacc_xh = tf.keras.layers.Input(shape=(W, 1))
lacc_yh = tf.keras.layers.Input(shape=(W, 1))
lacc_zh = tf.keras.layers.Input(shape=(W, 1))
gyr_xh = tf.keras.layers.Input(shape=(W, 1))
gyr_yh = tf.keras.layers.Input(shape=(W, 1))
gyr_zh = tf.keras.layers.Input(shape=(W, 1))
mag_xh = tf.keras.layers.Input(shape=(W, 1))
mag_yh = tf.keras.layers.Input(shape=(W, 1))
mag_zh = tf.keras.layers.Input(shape=(W, 1))
pressure_h = tf.keras.layers.Input(shape=(W, 1))

lacc_xl = tf.keras.layers.Input(shape=(W, 1))
lacc_yl = tf.keras.layers.Input(shape=(W, 1))
lacc_zl = tf.keras.layers.Input(shape=(W, 1))
gyr_xl = tf.keras.layers.Input(shape=(W, 1))
gyr_yl = tf.keras.layers.Input(shape=(W, 1))
gyr_zl = tf.keras.layers.Input(shape=(W, 1))
mag_xl = tf.keras.layers.Input(shape=(W, 1))
mag_yl = tf.keras.layers.Input(shape=(W, 1))
mag_zl = tf.keras.layers.Input(shape=(W, 1))
pressure_l = tf.keras.layers.Input(shape=(W, 1))

#teacher
#high_section
dropout_rate1_h = 0.1
def Conv1DKAN_block_h(X_h , Filters_h , grid_size_h ):
    u1_h , u2_h , u3_h , u4_h  = Filters_h 
    gs1_h , gs2_h , gs3_h , gs4_h  = grid_size_h 
    X_h  = DenseKAN(u1_h , gs1_h )(X_h )
    X_h  = tf.keras.layers.AveragePooling1D(pool_size = 4, padding = "same")(X_h)
    X_h  = tf.keras.layers.Dropout(dropout_rate1_h )(X_h)
    return X_h 

def ConvKAN_net_h(X_input_h ):
    X_h  = Conv1DKAN_block_h(X_input_h , Filters_h=[8, 8, 8, 8], grid_size_h=[10, 3, 3, 3])
    return X_h 

def ConvKAN_layer_h(lacc_xh, lacc_yh, lacc_zh, gyr_xh, gyr_yh, gyr_zh, mag_xh, mag_yh, mag_zh, pressure_h):
    lacc_xh1 = ConvKAN_net_h(lacc_xh)
    lacc_yh1 = ConvKAN_net_h(lacc_yh)
    lacc_zh1 = ConvKAN_net_h(lacc_zh)

    gyr_xh1 = ConvKAN_net_h(gyr_xh)
    gyr_yh1 = ConvKAN_net_h(gyr_yh)
    gyr_zh1 = ConvKAN_net_h(gyr_zh)

    mag_xh1 = ConvKAN_net_h(mag_xh)
    mag_yh1 = ConvKAN_net_h(mag_yh)
    mag_zh1 = ConvKAN_net_h(mag_zh)

    pressure_h1 = ConvKAN_net_h(pressure_h)

    cha_KAN_lacc_h  = tf.keras.layers.concatenate([lacc_xh1, lacc_yh1, lacc_zh1])
    cha_KAN_gyr_h  = tf.keras.layers.concatenate([gyr_xh1, gyr_yh1, gyr_zh1])
    cha_KAN_mag_h  = tf.keras.layers.concatenate([mag_xh1, mag_yh1, mag_zh1])
    cha_KAN_pre_h  =  tf.keras.layers.concatenate([pressure_h1])

    return cha_KAN_lacc_h , cha_KAN_gyr_h , cha_KAN_mag_h , cha_KAN_pre_h 
    
l_u_h = 8
l_gs_h = 10
def linked_KANlayer_h(cha_KAN_lacc_h , cha_KAN_gyr_h , cha_KAN_mag_h , cha_KAN_pre_h ):
    linked_KAN_lacc_h  =DenseKAN(l_u_h , l_gs_h )(cha_KAN_lacc_h )
    linked_KAN_gyr_h  = DenseKAN(l_u_h , l_gs_h )(cha_KAN_gyr_h )    
    linked_KAN_mag_h  = DenseKAN(l_u_h , l_gs_h )(cha_KAN_mag_h )
    linked_KAN_pre_h  = DenseKAN(l_u_h , l_gs_h )(cha_KAN_pre_h )    

    linked_KAN_h  = tf.keras.layers.concatenate([linked_KAN_lacc_h , linked_KAN_gyr_h , linked_KAN_mag_h , linked_KAN_pre_h ])
    return linked_KAN_h 

#low_section
dropout_rate1_l = 0.1
def Conv1DKAN_block_l(X_l , Filters_l , grid_size_l ):
    u1_l , u2_l , u3_l , u4_l  = Filters_l 
    gs1_l , gs2_l , gs3_l , gs4_l  = grid_size_l 

    X_l  = DenseKAN(u1_l , gs1_l )(X_l )
    X_l  = tf.keras.layers.AveragePooling1D(pool_size = 4, padding = "same")(X_l)
    X_l  = tf.keras.layers.Dropout(dropout_rate1_l )(X_l)

    return X_l 

def ConvKAN_net_l(X_input_l ):
    X_l  = Conv1DKAN_block_l(X_input_l , Filters_l=[8, 8, 8, 8], grid_size_l=[10, 3, 3, 3])
    return X_l 

def ConvKAN_layer_l(lacc_xh, lacc_yh, lacc_zh, gyr_xh, gyr_yh, gyr_zh, mag_xh, mag_yh, mag_zh, pressure_l):
    lacc_xl1 = ConvKAN_net_l(lacc_xl)
    lacc_yl1 = ConvKAN_net_l(lacc_yl)
    lacc_zl1 = ConvKAN_net_l(lacc_zl)

    gyr_xl1 = ConvKAN_net_l(gyr_xl)
    gyr_yl1 = ConvKAN_net_l(gyr_yl)
    gyr_zl1 = ConvKAN_net_l(gyr_zl)

    mag_xl1 = ConvKAN_net_l(mag_xl)
    mag_yl1 = ConvKAN_net_l(mag_yl)
    mag_zl1 = ConvKAN_net_l(mag_zl)

    pressure1 = ConvKAN_net_l(pressure_l)

    cha_KAN_lacc_l = tf.keras.layers.concatenate([lacc_xl1, lacc_yl1, lacc_zl1])
    cha_KAN_gyr_l = tf.keras.layers.concatenate([gyr_xl1, gyr_yl1, gyr_zl1])
    cha_KAN_mag_l = tf.keras.layers.concatenate([mag_xl1, mag_yl1, mag_zl1])
    cha_KAN_pre_l =  tf.keras.layers.concatenate([pressure1])
    return cha_KAN_lacc_l , cha_KAN_gyr_l , cha_KAN_mag_l , cha_KAN_pre_l 
    
l_u_l = 8
l_gs_l = 10
def linked_KANlayer_l(cha_KAN_lacc_l , cha_KAN_gyr_l , cha_KAN_mag_l , cha_KAN_pre_l ):
    linked_KAN_lacc_l  =DenseKAN(l_u_l , l_gs_l )(cha_KAN_lacc_l )
    linked_KAN_gyr_l  = DenseKAN(l_u_l , l_gs_l )(cha_KAN_gyr_l )
    linked_KAN_mag_l  = DenseKAN(l_u_l , l_gs_l )(cha_KAN_mag_l )
    linked_KAN_pre_l  = DenseKAN(l_u_l , l_gs_l )(cha_KAN_pre_l )    
    linked_KAN_l  = tf.keras.layers.concatenate([linked_KAN_lacc_l , linked_KAN_gyr_l , linked_KAN_mag_l , linked_KAN_pre_l ])
    return linked_KAN_l 

def merge(linked_KAN_h, linked_KAN_l):
    linked_KAN_h = tf.keras.layers.Reshape((125, 32, 1))(linked_KAN_h)
    linked_KAN_l = tf.keras.layers.Reshape((125, 32, 1))(linked_KAN_l)
    linked_KAN = tf.keras.layers.concatenate([linked_KAN_h , linked_KAN_l])
    linked_KAN_merge = tf.keras.layers.Dense(1)(linked_KAN)
    linked_KAN_merge = tf.keras.layers.Flatten()(linked_KAN_merge)
    return linked_KAN_merge

def mlp_layer(x):
    x = DenseKAN(8, grid_size = 10)(x)
    x = DenseKAN(8, grid_size = 10)(x)
    output = tf.keras.layers.Activation('softmax')(x)
    return output

cha_KAN_lacc_h, cha_KAN_gyr_h, cha_KAN_mag_h, cha_KAN_pre_h = ConvKAN_layer_h(
    lacc_xh, lacc_yh, lacc_zh, gyr_xh, gyr_yh, gyr_zh, mag_xh, mag_yh, mag_zh, pressure_h)
all_resnet_h = linked_KANlayer_h(cha_KAN_lacc_h, cha_KAN_gyr_h, cha_KAN_mag_h, cha_KAN_pre_h)
cha_KAN_lacc_l, cha_KAN_gyr_l, cha_KAN_mag_l, cha_KAN_pre_l = ConvKAN_layer_l(
    lacc_xh, lacc_yh, lacc_zh, mag_xh, gyr_xh, gyr_yh, gyr_zh, mag_yh, mag_zh, pressure_l)
all_resnet_l = linked_KANlayer_l(cha_KAN_lacc_l, cha_KAN_gyr_l, cha_KAN_mag_l, cha_KAN_pre_h)
all_resnet = merge(all_resnet_h, all_resnet_l)
output = mlp_layer(all_resnet)

teacher_model = tf.keras.Model(inputs=[
    lacc_xh, lacc_yh, lacc_zh, gyr_xh, gyr_yh, gyr_zh, mag_xh, mag_yh, mag_zh, pressure_h, 
    lacc_xl, lacc_yl, lacc_zl, gyr_xl, gyr_yl, gyr_zl, mag_xl, mag_yl, mag_zl, pressure_l
],outputs=output)

# Adam
initial_learning_rate = 1e-3
my_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
teacher_model.compile(optimizer=my_optimizer,  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
teacher_model.load_weights('best_model_teacher.h5')
teacher_model.trainable = False

#student
#student_high_section
dropout_rate1_h_s = 0.1
def Conv1DKAN_block_h_s(X_h_s , Filters_h_s , grid_size_h_s ):
    u1_h_s , u2_h_s , u3_h_s , u4_h_s  = Filters_h_s 
    gs1_h_s , gs2_h_s , gs3_h_s , gs4_h_s  = grid_size_h_s 
    X_h_s  = tf.keras.layers.Conv1D(u1_h_s, gs1_h_s, activation="relu", padding="same")(X_h_s )
    X_h_s  = tf.keras.layers.Conv1D(u2_h_s, gs2_h_s, activation="relu", padding="same")(X_h_s )
    X_h_s  = tf.keras.layers.Dropout(dropout_rate1_h_s )(X_h_s)
    return X_h_s 

def ConvKAN_net_h_s(X_input_h_s ):
    X_h_s  = Conv1DKAN_block_h_s(X_input_h_s , Filters_h_s=[3, 3, 8, 8], grid_size_h_s=[3, 3, 3, 3])
    return X_h_s 

def ConvKAN_layer_h_s(lacc_xh, lacc_yh, lacc_zh, gyr_xh, gyr_yh, gyr_zh, mag_xh, mag_yh, mag_zh, pressure_h_s):
    lacc_xh_s1 = ConvKAN_net_h_s(lacc_xh)
    lacc_yh_s1 = ConvKAN_net_h_s(lacc_yh)
    lacc_zh_s1 = ConvKAN_net_h_s(lacc_zh)

    gyr_xh_s1 = ConvKAN_net_h_s(gyr_xh)
    gyr_yh_s1 = ConvKAN_net_h_s(gyr_yh)
    gyr_zh_s1 = ConvKAN_net_h_s(gyr_zh)

    mag_xh_s1 = ConvKAN_net_h_s(mag_xh)
    mag_yh_s1 = ConvKAN_net_h_s(mag_yh)
    mag_zh_s1 = ConvKAN_net_h_s(mag_zh)

    pressure_h_s1 = ConvKAN_net_h_s(pressure_h_s)

    cha_KAN_lacc_h_s  = tf.keras.layers.concatenate([lacc_xh_s1, lacc_yh_s1, lacc_zh_s1])
    cha_KAN_gyr_h_s  = tf.keras.layers.concatenate([gyr_xh_s1, gyr_yh_s1, gyr_zh_s1])
    cha_KAN_mag_h_s  = tf.keras.layers.concatenate([mag_xh_s1, mag_yh_s1, mag_zh_s1])
    cha_KAN_pre_h_s  =  tf.keras.layers.concatenate([pressure_h_s1])

    return cha_KAN_lacc_h_s , cha_KAN_gyr_h_s , cha_KAN_mag_h_s , cha_KAN_pre_h_s 

l_u_h_s = 3  #linked_unites_high_student
l_gs_h_s = 3
def linked_KANlayer_h_s(cha_KAN_lacc_h_s , cha_KAN_gyr_h_s , cha_KAN_mag_h_s , cha_KAN_pre_h_s ):
    linked_KAN_lacc_h_s = tf.keras.layers.Conv1D(l_u_h_s, l_gs_h_s, activation="relu", padding="same")(cha_KAN_lacc_h_s )
    linked_KAN_gyr_h_s  = tf.keras.layers.Conv1D(l_u_h_s, l_gs_h_s, activation="relu", padding="same")(cha_KAN_gyr_h_s )  
    linked_KAN_mag_h_s  = tf.keras.layers.Conv1D(l_u_h_s, l_gs_h_s, activation="relu", padding="same")(cha_KAN_mag_h_s )
    linked_KAN_pre_h_s  = tf.keras.layers.Conv1D(l_u_h_s, l_gs_h_s, activation="relu", padding="same")(cha_KAN_pre_h_s )    
    linked_KAN_h_s  = tf.keras.layers.concatenate([linked_KAN_lacc_h_s , linked_KAN_gyr_h_s , linked_KAN_mag_h_s , linked_KAN_pre_h_s ])
    return linked_KAN_h_s  

#student_low_section
dropout_rate1_l_s = 0.1
def Conv1DKAN_block_l_s(X_l_s , Filters_l_s , grid_size_l_s ):
    u1_l_s , u2_l_s , u3_l_s , u4_l_s  = Filters_l_s 
    gs1_l_s , gs2_l_s , gs3_l_s , gs4_l_s  = grid_size_l_s 
    X_l_s  = tf.keras.layers.Conv1D(u1_l_s, gs1_l_s, activation="relu", padding="same")(X_l_s )
    X_l_s  = tf.keras.layers.Conv1D(u2_l_s, gs2_l_s, activation="relu", padding="same")(X_l_s )
    X_l_s  = tf.keras.layers.Dropout(dropout_rate1_l_s )(X_l_s)
    return X_l_s 

def ConvKAN_net_l_s(X_input_l_s ):
    X_l_s  = Conv1DKAN_block_l_s(X_input_l_s , Filters_l_s=[3, 8, 8, 8], grid_size_l_s=[3, 3, 3, 3])
    return X_l_s 

def ConvKAN_layer_l_s(lacc_xl, lacc_yl, lacc_zl, gyr_xl, gyr_yl, gyr_zl, mag_xl, mag_yl, mag_zl, pressure_l_s):
    lacc_xl_s1 = ConvKAN_net_l_s(lacc_xl)
    lacc_yl_s1 = ConvKAN_net_l_s(lacc_yl)
    lacc_zl_s1 = ConvKAN_net_l_s(lacc_zl)

    gyr_xl_s1 = ConvKAN_net_l_s(gyr_xl)
    gyr_yl_s1 = ConvKAN_net_l_s(gyr_yl)
    gyr_zl_s1 = ConvKAN_net_l_s(gyr_zl)

    mag_xl_s1 = ConvKAN_net_l_s(mag_xl)
    mag_yl_s1 = ConvKAN_net_l_s(mag_yl)
    mag_zl_s1 = ConvKAN_net_l_s(mag_zl)

    pressure_l_s1 = ConvKAN_net_l_s(pressure_l_s)

    cha_KAN_lacc_l_s  = tf.keras.layers.concatenate([lacc_xl_s1, lacc_yl_s1, lacc_zl_s1])
    cha_KAN_gyr_l_s  = tf.keras.layers.concatenate([gyr_xl_s1, gyr_yl_s1, gyr_zl_s1])
    cha_KAN_mag_l_s  = tf.keras.layers.concatenate([mag_xl_s1, mag_yl_s1, mag_zl_s1])
    cha_KAN_pre_l_s  =  tf.keras.layers.concatenate([pressure_l_s1])

    return cha_KAN_lacc_l_s , cha_KAN_gyr_l_s , cha_KAN_mag_l_s , cha_KAN_pre_l_s 

l_u_l_s = 3  #linked_unites_low_student
l_gs_l_s = 3
def linked_KANlayer_l_s(cha_KAN_lacc_l_s , cha_KAN_gyr_l_s , cha_KAN_mag_l_s , cha_KAN_pre_l_s ):
    linked_KAN_lacc_l_s = tf.keras.layers.Conv1D(l_u_l_s, l_gs_l_s, activation="relu", padding="same")(cha_KAN_lacc_l_s )
    linked_KAN_gyr_l_s  = tf.keras.layers.Conv1D(l_u_l_s, l_gs_l_s, activation="relu", padding="same")(cha_KAN_gyr_l_s )
    linked_KAN_mag_l_s  = tf.keras.layers.Conv1D(l_u_l_s, l_gs_l_s, activation="relu", padding="same")(cha_KAN_mag_l_s )
    linked_KAN_pre_l_s  = tf.keras.layers.Conv1D(l_u_l_s, l_gs_l_s, activation="relu", padding="same")(cha_KAN_pre_l_s )    
    linked_KAN_l_s  = tf.keras.layers.concatenate([linked_KAN_lacc_l_s , linked_KAN_gyr_l_s , linked_KAN_mag_l_s , linked_KAN_pre_l_s ])
    return linked_KAN_l_s  

def merge_s(linked_KAN_h_s, linked_KAN_l_s):
    linked_KAN_h_s = tf.keras.layers.Reshape((500, 12, 1))(linked_KAN_h_s)
    linked_KAN_l_s = tf.keras.layers.Reshape((500, 12, 1))(linked_KAN_l_s)
    linked_KAN = tf.keras.layers.concatenate([linked_KAN_h_s , linked_KAN_l_s])
    linked_KAN_merge = tf.keras.layers.Dense(1)(linked_KAN)
    linked_KAN_merge = tf.keras.layers.Flatten()(linked_KAN_merge)
    return linked_KAN_merge

def mlp_layer_s(x):
    x = tf.keras.layers.Dense(8)(x)
    output = tf.keras.layers.Activation('softmax')(x)
    return output

cha_KAN_lacc_h, cha_KAN_gyr_h, cha_KAN_mag_h, cha_KAN_pre_h = ConvKAN_layer_h_s(
    lacc_xh, lacc_yh, lacc_zh, gyr_xh, gyr_yh, gyr_zh, mag_xh, mag_yh, mag_zh, pressure_h)
all_resnet_h = linked_KANlayer_h_s(cha_KAN_lacc_h, cha_KAN_gyr_h, cha_KAN_mag_h, cha_KAN_pre_h)
cha_KAN_lacc_l, cha_KAN_gyr_l, cha_KAN_mag_l, cha_KAN_pre_l = ConvKAN_layer_l_s(
    lacc_xh, lacc_yh, lacc_zh, mag_xh, gyr_xh, gyr_yh, gyr_zh, mag_yh, mag_zh, pressure_l)
all_resnet_l = linked_KANlayer_l_s(cha_KAN_lacc_l, cha_KAN_gyr_l, cha_KAN_mag_l, cha_KAN_pre_h)
all_resnet = merge_s(all_resnet_h, all_resnet_l)
output = mlp_layer_s(all_resnet)


student_model = tf.keras.Model(inputs=[
    lacc_xh, lacc_yh, lacc_zh, gyr_xh, gyr_yh, gyr_zh, mag_xh, mag_yh, mag_zh, pressure_h, 
    lacc_xl, lacc_yl, lacc_zl, gyr_xl, gyr_yl, gyr_zl, mag_xl, mag_yl, mag_zl, pressure_l
],outputs=output)
student_model.summary()

#Distiller
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            tf.keras.activations.softmax(teacher_pred / self.temperature, axis=1),     
            tf.keras.activations.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)



distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=5,
)

checkpoint_path = "training_2/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='sparse_categorical_accuracy', 
    patience=20, 
    mode='max', 
    verbose=1
)
# ReduceLROnPlateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='sparse_categorical_accuracy', 
    factor=0.2, 
    patience=5, 
    min_lr=1e-3,
    verbose=1
)

distiller.fit(
    [train_xh[:, :, 0], train_xh[:, :, 1], train_xh[:, :, 2], train_xh[:, :, 3],train_xh[:, :, 4],
     train_xh[:, :, 5], train_xh[:, :, 6], train_xh[:, :, 7], train_xh[:, :, 8], train_xh[:, :, 9],
     train_xl[:, :, 0], train_xl[:, :, 1], train_xl[:, :, 2], train_xl[:, :, 3],train_xl[:, :, 4],
     train_xl[:, :, 5], train_xl[:, :, 6], train_xl[:, :, 7], train_xl[:, :, 8], train_xl[:, :, 9]],
    train_y, validation_data=(
    [val_xh[:, :, 0], val_xh[:, :, 1], val_xh[:, :, 2], val_xh[:, :, 3], val_xh[:, :, 4], 
    val_xh[:, :, 5], val_xh[:, :, 6], val_xh[:, :, 7], val_xh[:, :, 8], val_xh[:, :, 9],
    val_xl[:, :, 0], val_xl[:, :, 1], val_xl[:, :, 2], val_xl[:, :, 3], val_xl[:, :, 4], 
    val_xl[:, :, 5], val_xl[:, :, 6], val_xl[:, :, 7], val_xl[:, :, 8], val_xl[:, :, 9]],
    val_y), epochs = 150, shuffle= True , batch_size = 256, callbacks = [early_stopping_callback, reduce_lr])


predictions = student_model.predict([
    val_xh[:, :, 0], val_xh[:, :, 1], val_xh[:, :, 2], val_xh[:, :, 3], val_xh[:, :, 4], 
    val_xh[:, :, 5], val_xh[:, :, 6], val_xh[:, :, 7], val_xh[:, :, 8], val_xh[:, :, 9],
    val_xl[:, :, 0], val_xl[:, :, 1], val_xl[:, :, 2], val_xl[:, :, 3], val_xl[:, :, 4], 
    val_xl[:, :, 5], val_xl[:, :, 6], val_xl[:, :, 7], val_xl[:, :, 8], val_xl[:, :, 9]
    ])

predictions = [np.argmax(p) for p in predictions]
lv = np.reshape(val_y, newshape=-1)
accuracy = 0
cnf = [[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]]
for (p, t) in zip(predictions, lv):
  p = int(p)
  t = int(t)
  cnf[t][p] += 1
  if p == t:
    accuracy += 1
accuracy /= float(len(predictions))
print('acc', accuracy)
print(np.array(cnf))
print('1: %f\n2: %f\n3: %f\n4: %f\n5: %f\n6: %f\n7: %f\n8: %f\n' % (cnf[0][0] / float(np.sum(cnf[0])),
                                                                    cnf[1][1] / float(np.sum(cnf[1])),
                                                                    cnf[2][2] / float(np.sum(cnf[2])),
                                                                    cnf[3][3] / float(np.sum(cnf[3])),
                                                                    cnf[4][4] / float(np.sum(cnf[4])),
                                                                  cnf[5][5] / float(np.sum(cnf[5])),
                                                                  cnf[6][6] / float(np.sum(cnf[6])),
                                                                  cnf[7][7] / float(np.sum(cnf[7])),
                                                                    ))


