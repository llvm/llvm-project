# Conversion operations

vsetvli x28, x0, e16, mf2, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e16, m8, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e32, m8, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e64, m1, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vsext.vf2 v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vsext.vf2 v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e16, m8, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e32, m8, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e64, m1, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vzext.vf2 v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vzext.vf2 v8, v16

vsetvli x28, x0, e32, mf2, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e32, m8, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e64, m1, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vsext.vf4 v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vsext.vf4 v8, v16

vsetvli x28, x0, e32, mf2, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e32, m8, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e64, m1, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vzext.vf4 v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vzext.vf4 v8, v16

vsetvli x28, x0, e64, m1, tu, mu
vsext.vf8 v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vsext.vf8 v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vsext.vf8 v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vsext.vf8 v8, v16

vsetvli x28, x0, e64, m1, tu, mu
vzext.vf8 v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vzext.vf8 v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vzext.vf8 v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vzext.vf8 v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfcvt.f.xu.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfcvt.f.xu.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfcvt.f.x.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfcvt.f.x.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfcvt.rtz.x.f.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfcvt.rtz.x.f.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfcvt.rtz.xu.f.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfcvt.rtz.xu.f.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfcvt.x.f.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfcvt.x.f.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfcvt.xu.f.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfcvt.xu.f.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.f.f.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.f.f.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.f.xu.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.f.xu.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.f.x.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.f.x.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.rod.f.f.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.rod.f.f.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.rtz.x.f.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.rtz.x.f.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.rtz.xu.f.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.rtz.xu.f.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.x.f.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.x.f.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfncvt.xu.f.w v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfncvt.xu.f.w v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfwcvt.f.f.v v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfwcvt.f.f.v v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfwcvt.f.x.v v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfwcvt.f.x.v v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfwcvt.f.xu.v v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfwcvt.f.xu.v v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfwcvt.rtz.x.f.v v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfwcvt.rtz.x.f.v v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfwcvt.rtz.xu.f.v v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfwcvt.rtz.xu.f.v v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfwcvt.x.f.v v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfwcvt.x.f.v v8, v16

vsetvli x28, x0, e16, mf2, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vfwcvt.xu.f.v v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vfwcvt.xu.f.v v8, v16
