# Permutation and shuffle operations

vsetvli x28, x0, e8, mf2, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmv.v.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmv.v.v v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e8, mf4, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e8, mf8, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e8, m1, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e8, m2, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e8, m4, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e8, m8, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e16, mf2, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e16, mf4, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e16, m1, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e16, m2, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e16, m4, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e16, m8, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e32, mf2, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e32, m1, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e32, m2, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e32, m4, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e32, m8, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e64, m1, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e64, m2, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e64, m4, tu, mu
vmv.v.x v8, x8
vsetvli x28, x0, e64, m8, tu, mu
vmv.v.x v8, x8

vsetvli x28, x0, e8, mf2, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e8, mf4, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e8, mf8, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e8, m1, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e8, m2, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e8, m4, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e8, m8, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e16, mf2, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e16, mf4, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e16, m1, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e16, m2, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e16, m4, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e16, m8, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e32, mf2, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e32, m1, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e32, m2, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e32, m4, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e32, m8, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e64, m1, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e64, m2, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e64, m4, tu, mu
vmv.v.i v8, 12
vsetvli x28, x0, e64, m8, tu, mu
vmv.v.i v8, 12

vsetvli x28, x0, e8, mf2, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmv.x.s x8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmv.x.s x8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e8, mf4, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e8, mf8, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e8, m1, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e8, m2, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e8, m4, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e8, m8, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e16, mf2, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e16, mf4, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e16, m1, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e16, m2, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e16, m4, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e16, m8, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e32, mf2, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e32, m1, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e32, m2, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e32, m4, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e32, m8, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e64, m1, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e64, m2, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e64, m4, tu, mu
vmv.s.x v8, x8
vsetvli x28, x0, e64, m8, tu, mu
vmv.s.x v8, x8

vsetvli x28, x0, e8, mf2, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmv1r.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmv1r.v v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmv2r.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmv2r.v v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmv4r.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmv4r.v v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmv8r.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmv8r.v v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
viota.m v8, v16
vsetvli x28, x0, e8, mf4, tu, mu
viota.m v8, v16
vsetvli x28, x0, e8, mf8, tu, mu
viota.m v8, v16
vsetvli x28, x0, e8, m1, tu, mu
viota.m v8, v16
vsetvli x28, x0, e8, m2, tu, mu
viota.m v8, v16
vsetvli x28, x0, e8, m4, tu, mu
viota.m v8, v16
vsetvli x28, x0, e8, m8, tu, mu
viota.m v8, v16
vsetvli x28, x0, e16, mf2, tu, mu
viota.m v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
viota.m v8, v16
vsetvli x28, x0, e16, m1, tu, mu
viota.m v8, v16
vsetvli x28, x0, e16, m2, tu, mu
viota.m v8, v16
vsetvli x28, x0, e16, m4, tu, mu
viota.m v8, v16
vsetvli x28, x0, e16, m8, tu, mu
viota.m v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
viota.m v8, v16
vsetvli x28, x0, e32, m1, tu, mu
viota.m v8, v16
vsetvli x28, x0, e32, m2, tu, mu
viota.m v8, v16
vsetvli x28, x0, e32, m4, tu, mu
viota.m v8, v16
vsetvli x28, x0, e32, m8, tu, mu
viota.m v8, v16
vsetvli x28, x0, e64, m1, tu, mu
viota.m v8, v16
vsetvli x28, x0, e64, m2, tu, mu
viota.m v8, v16
vsetvli x28, x0, e64, m4, tu, mu
viota.m v8, v16
vsetvli x28, x0, e64, m8, tu, mu
viota.m v8, v16

vsetvli x28, x0, e8, mf2, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e8, m8, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e16, m8, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e32, m8, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e64, m1, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e64, m2, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e64, m4, tu, mu
vcompress.vm v8, v16, v24
vsetvli x28, x0, e64, m8, tu, mu
vcompress.vm v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e8, m8, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e16, m8, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e32, m8, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e64, m1, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e64, m2, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e64, m4, tu, mu
vslide1up.vx v8, v16, x30
vsetvli x28, x0, e64, m8, tu, mu
vslide1up.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e8, m8, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e16, m8, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e32, m8, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e64, m1, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e64, m2, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e64, m4, tu, mu
vslide1down.vx v8, v16, x30
vsetvli x28, x0, e64, m8, tu, mu
vslide1down.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e8, m8, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e16, m8, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e32, m8, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e64, m1, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e64, m2, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e64, m4, tu, mu
vslideup.vx v8, v16, x30
vsetvli x28, x0, e64, m8, tu, mu
vslideup.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e8, mf4, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e8, mf8, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e8, m1, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e8, m2, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e8, m4, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e8, m8, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e16, mf2, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e16, mf4, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e16, m1, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e16, m2, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e16, m4, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e16, m8, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e32, mf2, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e32, m1, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e32, m2, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e32, m4, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e32, m8, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e64, m1, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e64, m2, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e64, m4, tu, mu
vslideup.vi v8, v16, 12
vsetvli x28, x0, e64, m8, tu, mu
vslideup.vi v8, v16, 12

vsetvli x28, x0, e8, mf2, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e8, m8, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e16, m8, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e32, m8, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e64, m1, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e64, m2, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e64, m4, tu, mu
vslidedown.vx v8, v16, x30
vsetvli x28, x0, e64, m8, tu, mu
vslidedown.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e8, mf4, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e8, mf8, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e8, m1, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e8, m2, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e8, m4, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e8, m8, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e16, mf2, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e16, mf4, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e16, m1, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e16, m2, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e16, m4, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e16, m8, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e32, mf2, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e32, m1, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e32, m2, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e32, m4, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e32, m8, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e64, m1, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e64, m2, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e64, m4, tu, mu
vslidedown.vi v8, v16, 12
vsetvli x28, x0, e64, m8, tu, mu
vslidedown.vi v8, v16, 12

vsetvli x28, x0, e8, mf2, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e8, m8, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e16, m8, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e32, m8, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e64, m1, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e64, m2, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e64, m4, tu, mu
vrgather.vv v8, v16, v24
vsetvli x28, x0, e64, m8, tu, mu
vrgather.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e8, m8, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e16, m8, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e32, m8, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e64, m1, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e64, m2, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e64, m4, tu, mu
vrgather.vx v8, v16, x30
vsetvli x28, x0, e64, m8, tu, mu
vrgather.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e8, mf4, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e8, mf8, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e8, m1, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e8, m2, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e8, m4, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e8, m8, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e16, mf2, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e16, mf4, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e16, m1, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e16, m2, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e16, m4, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e16, m8, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e32, mf2, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e32, m1, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e32, m2, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e32, m4, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e32, m8, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e64, m1, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e64, m2, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e64, m4, tu, mu
vrgather.vi v8, v16, 12
vsetvli x28, x0, e64, m8, tu, mu
vrgather.vi v8, v16, 12

vsetvli x28, x0, e8, mf2, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vrgatherei16.vv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e16, m8, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e32, m8, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e64, m1, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e64, m2, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e64, m4, tu, mu
vrgatherei16.vv v8, v16, v24
vsetvli x28, x0, e64, m8, tu, mu
vrgatherei16.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e8, mf4, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e8, mf8, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e8, m1, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e8, m2, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e8, m4, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e8, m8, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e16, mf2, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e16, mf4, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e16, m1, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e16, m2, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e16, m4, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e16, m8, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e32, mf2, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e32, m1, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e32, m2, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e32, m4, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e32, m8, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e64, m1, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e64, m2, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e64, m4, tu, mu
vmerge.vim v8, v8, 12, v0
vsetvli x28, x0, e64, m8, tu, mu
vmerge.vim v8, v8, 12, v0

vsetvli x28, x0, e8, mf2, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e8, mf4, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e8, mf8, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e8, m1, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e8, m2, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e8, m4, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e8, m8, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e16, mf2, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e16, mf4, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e16, m1, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e16, m2, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e16, m4, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e16, m8, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e32, mf2, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e32, m1, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e32, m2, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e32, m4, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e32, m8, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e64, m1, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e64, m2, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e64, m4, tu, mu
vmerge.vvm v8, v8, v8, v0
vsetvli x28, x0, e64, m8, tu, mu
vmerge.vvm v8, v8, v8, v0

vsetvli x28, x0, e8, mf2, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e8, mf4, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e8, mf8, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e8, m1, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e8, m2, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e8, m4, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e8, m8, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e16, mf2, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e16, mf4, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e16, m1, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e16, m2, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e16, m4, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e16, m8, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e32, mf2, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e32, m1, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e32, m2, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e32, m4, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e32, m8, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e64, m1, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e64, m2, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e64, m4, tu, mu
vmerge.vxm v8, v8, x30, v0
vsetvli x28, x0, e64, m8, tu, mu
vmerge.vxm v8, v8, x30, v0

vsetvli x28, x0, e16, mf2, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e16, mf4, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e16, m1, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e16, m2, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e16, m4, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e16, m8, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e32, mf2, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e32, m1, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e32, m2, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e32, m4, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e32, m8, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e64, m1, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e64, m2, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e64, m4, tu, mu
vfmerge.vfm v8, v8, ft0, v0
vsetvli x28, x0, e64, m8, tu, mu
vfmerge.vfm v8, v8, ft0, v0

vsetvli x28, x0, e16, mf2, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfslide1down.vf v8, v16, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfslide1down.vf v8, v16, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfslide1up.vf v8, v16, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfslide1up.vf v8, v16, ft0
