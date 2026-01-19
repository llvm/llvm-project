# Min/max operations

vsetvli x28, x0, e8, mf2, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmax.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmax.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vmax.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vmax.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmaxu.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmaxu.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vmaxu.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vmaxu.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmin.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmin.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vmin.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vmin.vx v8, v8, x30

vsetvli x28, x0, e8, mf2, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vminu.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vminu.vv v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e8, mf4, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e8, mf8, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e8, m1, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e8, m2, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e8, m4, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e8, m8, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e16, mf2, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e16, mf4, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e16, m1, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e16, m2, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e16, m4, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e16, m8, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e32, mf2, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e32, m1, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e32, m2, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e32, m4, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e32, m8, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e64, m1, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e64, m2, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e64, m4, tu, mu
vminu.vx v8, v8, x30
vsetvli x28, x0, e64, m8, tu, mu
vminu.vx v8, v8, x30
