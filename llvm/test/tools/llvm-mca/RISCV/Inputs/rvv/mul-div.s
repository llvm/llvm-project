# Multiplication and division operations

vsetvli x28, x0, e8, mf2, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmul.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmul.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vmul.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vmul.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vdiv.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vdiv.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vdiv.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vdivu.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vdivu.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vdivu.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vdivu.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vrem.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vrem.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vrem.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vrem.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vremu.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vremu.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vremu.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vremu.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmulh.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmulh.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vmulh.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vmulh.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmulhu.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmulhu.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vmulhu.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vmulhu.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmulhsu.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmulhsu.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vmulhsu.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vmulhsu.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmul.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmul.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vwmul.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vwmul.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmulu.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmulu.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vwmulu.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vwmulu.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e8, mf4, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e8, mf8, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e8, m1, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e8, m2, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e8, m4, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e16, mf2, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vwmulsu.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vwmulsu.vv v8, v16, v24

vsetvli x28, x0, e8, mf2, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e8, mf4, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e8, mf8, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e8, m1, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e8, m2, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e8, m4, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e16, mf2, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e16, mf4, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e16, m1, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e16, m2, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e16, m4, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e32, mf2, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e32, m1, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e32, m2, tu, mu
vwmulsu.vx v8, v16, x30
vsetvli x28, x0, e32, m4, tu, mu
vwmulsu.vx v8, v16, x30

vsetvli x28, x0, e8, mf2, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vsmul.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vsmul.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vsmul.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vsmul.vx v8, v8, x30
