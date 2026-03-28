# Fused multiply-add operations

vsetvli x28, x0, e8, mf2, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmacc.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmacc.vx v8, x8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmacc.vx v8, x8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmadd.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmadd.vx v8, x8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmadd.vx v8, x8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vnmsac.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e8, m1, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e8, m2, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e8, m4, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e8, m8, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e16, m1, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e16, m2, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e16, m4, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e16, m8, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e32, m1, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e32, m2, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e32, m4, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e32, m8, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e64, m1, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e64, m2, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e64, m4, tu, mu
vnmsac.vx v8, x8, v8
vsetvli x28, x0, e64, m8, tu, mu
vnmsac.vx v8, x8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vnmsub.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e8, m1, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e8, m2, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e8, m4, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e8, m8, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e16, m1, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e16, m2, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e16, m4, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e16, m8, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e32, m1, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e32, m2, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e32, m4, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e32, m8, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e64, m1, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e64, m2, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e64, m4, tu, mu
vnmsub.vx v8, x8, v8
vsetvli x28, x0, e64, m8, tu, mu
vnmsub.vx v8, x8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmaccu.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmaccu.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmaccu.vx v8, x16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmaccu.vx v8, x16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmacc.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmacc.vx v8, x16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmacc.vx v8, x16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmaccsu.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmaccsu.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmaccsu.vx v8, x16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmaccsu.vx v8, x16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmaccus.vx v8, x16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmaccus.vx v8, x16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwmacc.vf v8, f16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwmacc.vf v8, f16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwmacc.vv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwmsac.vf v8, f16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwmsac.vf v8, f16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwmsac.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwmsac.vv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwnmacc.vf v8, f16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwnmacc.vf v8, f16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwnmacc.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwnmacc.vv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwnmsac.vf v8, f16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwnmsac.vf v8, f16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwnmsac.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwnmsac.vv v8, v16, v24
