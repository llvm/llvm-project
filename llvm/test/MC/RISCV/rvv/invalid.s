# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-v --mattr=+f %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsetvli a2, a0, e31
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e32,m3
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, m1,e32
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e32,m16
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e2048,m8
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e1,m8
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,m1,tx
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,m1,ta,mx
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,m1,ma
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,m1,mu
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8x,m1,tu,mu
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,m1z,tu,mu
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,mf1,tu,mu
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,m1,tu,mut
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vsetvli a2, a0, e8,m1,tut,mu
# CHECK-ERROR: operand must be e[8|16|32|64|128|256|512|1024],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]

vadd.vv v1, v3, v2, v4.t
# CHECK-ERROR: operand must be v0.t

vadd.vv v1, v3, v2, v0
# CHECK-ERROR: expected '.t' suffix

vadd.vv v1, v3, a0
# CHECK-ERROR: invalid operand for instruction

vmslt.vi v1, v2, -16
# CHECK-ERROR: immediate must be in the range [-15, 16]

vmslt.vi v1, v2, 17
# CHECK-ERROR: immediate must be in the range [-15, 16]

viota.m v0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: viota.m v0, v2, v0.t

viota.m v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: viota.m v2, v2

vfwcvt.xu.f.v v0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwcvt.xu.f.v v0, v2, v0.t

vfwcvt.xu.f.v v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwcvt.xu.f.v v2, v2

vfwcvt.x.f.v v0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwcvt.x.f.v v0, v2, v0.t

vfwcvt.x.f.v v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwcvt.x.f.v v2, v2

vfwcvt.f.xu.v v0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwcvt.f.xu.v v0, v2, v0.t

vfwcvt.f.xu.v v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwcvt.f.xu.v v2, v2

vfwcvt.f.x.v v0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwcvt.f.x.v v0, v2, v0.t

vfwcvt.f.x.v v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwcvt.f.x.v v2, v2

vfwcvt.f.f.v v0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwcvt.f.f.v v0, v2, v0.t

vfwcvt.f.f.v v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwcvt.f.f.v v2, v2

vslideup.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vslideup.vx v0, v2, a0, v0.t

vslideup.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vslideup.vx v2, v2, a0

vslideup.vi v0, v2, 31, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vslideup.vi v0, v2, 31, v0.t

vslideup.vi v2, v2, 31
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vslideup.vi v2, v2, 31

vslide1up.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vslide1up.vx v0, v2, a0, v0.t

vslide1up.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vslide1up.vx v2, v2, a0

vnsrl.wv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnsrl.wv v2, v2, v4

vnsrl.wx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnsrl.wx v2, v2, a0

vnsrl.wi v2, v2, 31
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnsrl.wi v2, v2, 31

vnsra.wv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnsra.wv v2, v2, v4

vnsra.wx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnsra.wx v2, v2, a0

vnsra.wi v2, v2, 31
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnsra.wi v2, v2, 31

vnclipu.wv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnclipu.wv v2, v2, v4

vnclipu.wx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnclipu.wx v2, v2, a0

vnclipu.wi v2, v2, 31
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnclipu.wi v2, v2, 31

vnclip.wv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnclip.wv v2, v2, v4

vnclip.wx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnclip.wx v2, v2, a0

vnclip.wi v2, v2, 31
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vnclip.wi v2, v2, 31

vfncvt.xu.f.w v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfncvt.xu.f.w v2, v2

vfncvt.x.f.w v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfncvt.x.f.w v2, v2

vfncvt.f.xu.w v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfncvt.f.xu.w v2, v2

vfncvt.f.x.w v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfncvt.f.x.w v2, v2

vfncvt.f.f.w v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfncvt.f.f.w v2, v2

vfncvt.rod.f.f.w v2, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfncvt.rod.f.f.w v2, v2

vrgather.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vrgather.vv v0, v2, v4, v0.t

vrgather.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vrgather.vv v2, v2, v4

vrgather.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vrgather.vx v0, v2, a0, v0.t

vrgather.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vrgather.vx v2, v2, a0

vrgather.vi v0, v2, 31, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vrgather.vi v0, v2, 31, v0.t

vrgather.vi v2, v2, 31
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vrgather.vi v2, v2, 31

vwaddu.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwaddu.vv v0, v2, v4, v0.t

vwaddu.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwaddu.vv v2, v2, v4

vwsubu.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsubu.vv v0, v2, v4, v0.t

vwsubu.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsubu.vv v2, v2, v4

vwadd.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwadd.vv v0, v2, v4, v0.t

vwadd.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwadd.vv v2, v2, v4

vwsub.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsub.vv v0, v2, v4, v0.t

vwsub.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsub.vv v2, v2, v4

vwmul.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmul.vv v0, v2, v4, v0.t

vwmul.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmul.vv v2, v2, v4

vwmulu.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmulu.vv v0, v2, v4, v0.t

vwmulu.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmulu.vv v2, v2, v4

vwmulsu.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmulsu.vv v0, v2, v4, v0.t

vwmulsu.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmulsu.vv v2, v2, v4

vwmaccu.vv v0, v4, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmaccu.vv v0, v4, v2, v0.t

vwmaccu.vv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmaccu.vv v2, v4, v2

vwmacc.vv v0, v4, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmacc.vv v0, v4, v2, v0.t

vwmacc.vv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmacc.vv v2, v4, v2

vwmaccsu.vv v0, v4, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmaccsu.vv v0, v4, v2, v0.t

vwmaccsu.vv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmaccsu.vv v2, v4, v2

vfwadd.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwadd.vv v0, v2, v4, v0.t

vfwadd.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwadd.vv v2, v2, v4

vfwsub.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwsub.vv v0, v2, v4, v0.t

vfwsub.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwsub.vv v2, v2, v4

vfwmul.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwmul.vv v0, v2, v4, v0.t

vfwmul.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwmul.vv v2, v2, v4

vfwmacc.vv v0, v4, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwmacc.vv v0, v4, v2, v0.t

vfwmacc.vv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwmacc.vv v2, v4, v2

vfwnmacc.vv v0, v4, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwnmacc.vv v0, v4, v2, v0.t

vfwnmacc.vv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwnmacc.vv v2, v4, v2

vfwmsac.vv v0, v4, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwmsac.vv v0, v4, v2, v0.t

vfwmsac.vv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwmsac.vv v2, v4, v2

vfwnmsac.vv v0, v4, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwnmsac.vv v0, v4, v2, v0.t

vfwnmsac.vv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwnmsac.vv v2, v4, v2

vwaddu.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwaddu.vx v0, v2, a0, v0.t

vwaddu.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwaddu.vx v2, v2, a0

vwsubu.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsubu.vx v0, v2, a0, v0.t

vwsubu.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsubu.vx v2, v2, a0

vwadd.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwadd.vx v0, v2, a0, v0.t

vwadd.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwadd.vx v2, v2, a0

vwsub.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsub.vx v0, v2, a0, v0.t

vwsub.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsub.vx v2, v2, a0

vwmul.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmul.vx v0, v2, a0, v0.t

vwmul.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmul.vx v2, v2, a0

vwmulu.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmulu.vx v0, v2, a0, v0.t

vwmulu.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmulu.vx v2, v2, a0

vwmulsu.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmulsu.vx v0, v2, a0, v0.t

vwmulsu.vx v2, v2, a0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmulsu.vx v2, v2, a0

vwmaccu.vx v0, a0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmaccu.vx v0, a0, v2, v0.t

vwmaccu.vx v2, a0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmaccu.vx v2, a0, v2

vwmacc.vx v0, a0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmacc.vx v0, a0, v2, v0.t

vwmacc.vx v2, a0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmacc.vx v2, a0, v2

vwmaccsu.vx v0, a0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmaccsu.vx v0, a0, v2, v0.t

vwmaccsu.vx v2, a0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmaccsu.vx v2, a0, v2

vwmaccus.vx v0, a0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwmaccus.vx v0, a0, v2, v0.t

vwmaccus.vx v2, a0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwmaccus.vx v2, a0, v2

vfwadd.vf v0, v2, fa0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwadd.vf v0, v2, fa0, v0.t

vfwadd.vf v2, v2, fa0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwadd.vf v2, v2, fa0

vfwsub.vf v0, v2, fa0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwsub.vf v0, v2, fa0, v0.t

vfwsub.vf v2, v2, fa0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwsub.vf v2, v2, fa0

vfwmul.vf v0, v2, fa0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwmul.vf v0, v2, fa0, v0.t

vfwmul.vf v2, v2, fa0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwmul.vf v2, v2, fa0

vfwmacc.vf v0, fa0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwmacc.vf v0, fa0, v2, v0.t

vfwmacc.vf v2, fa0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwmacc.vf v2, fa0, v2

vfwnmacc.vf v0, fa0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwnmacc.vf v0, fa0, v2, v0.t

vfwnmacc.vf v2, fa0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwnmacc.vf v2, fa0, v2

vfwmsac.vf v0, fa0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwmsac.vf v0, fa0, v2, v0.t

vfwmsac.vf v2, fa0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwmsac.vf v2, fa0, v2

vfwnmsac.vf v0, fa0, v2, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwnmsac.vf v0, fa0, v2, v0.t

vfwnmsac.vf v2, fa0, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwnmsac.vf v2, fa0, v2

vcompress.vm v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vcompress.vm v2, v2, v4

vwaddu.wv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwaddu.wv v0, v2, v4, v0.t

vwaddu.wv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwaddu.wv v2, v4, v2

vwsubu.wv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsubu.wv v0, v2, v4, v0.t

vwsubu.wv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsubu.wv v2, v4, v2

vwadd.wv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwadd.wv v0, v2, v4, v0.t

vwadd.wv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwadd.wv v2, v4, v2

vwsub.wv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsub.wv v0, v2, v4, v0.t

vwsub.wv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsub.wv v2, v4, v2

vfwadd.wv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwadd.wv v0, v2, v4, v0.t

vfwadd.wv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwadd.wv v2, v4, v2

vfwsub.wv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwsub.wv v0, v2, v4, v0.t

vfwsub.wv v2, v4, v2
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vfwsub.wv v2, v4, v2

vwaddu.wx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwaddu.wx v0, v2, a0, v0.t

vwsubu.wx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsubu.wx v0, v2, a0, v0.t

vwadd.wx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwadd.wx v0, v2, a0, v0.t

vwsub.wx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vwsub.wx v0, v2, a0, v0.t

vfwadd.wf v0, v2, fa0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwadd.wf v0, v2, fa0, v0.t

vfwsub.wf v0, v2, fa0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfwsub.wf v0, v2, fa0, v0.t

vadc.vvm v0, v2, v4, v0
# CHECK-ERROR: The destination vector register group cannot be V0.
# CHECK-ERROR-LABEL: vadc.vvm v0, v2, v4, v0

vmadc.vvm v2, v2, v4, v0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vmadc.vvm v2, v2, v4, v0

vmadc.vvm v4, v2, v4, v0
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vmadc.vvm v4, v2, v4, v0

vadd.vv v0, v2, v4, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vadd.vv v0, v2, v4, v0.t

vadd.vx v0, v2, a0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vadd.vx v0, v2, a0, v0.t

vadd.vi v0, v2, 1, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vadd.vi v0, v2, 1, v0.t

vmsge.vx v0, v4, a0, v0.t
# CHECK-ERROR: too few operands for instruction
# CHECK-ERROR-LABEL: vmsge.vx v0, v4, a0, v0.t

vmsge.vx v8, v4, a0, v0.t, v2
# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: vmsge.vx v8, v4, a0, v0.t, v2

vmerge.vim v0, v1, 1, v0
# CHECK-ERROR: The destination vector register group cannot be V0.
# CHECK-ERROR-LABEL: vmerge.vim v0, v1, 1, v0

vmerge.vvm v0, v1, v2, v0
# CHECK-ERROR: The destination vector register group cannot be V0.
# CHECK-ERROR-LABEL: vmerge.vvm v0, v1, v2, v0

vmerge.vxm v0, v1, x1, v0
# CHECK-ERROR: The destination vector register group cannot be V0.
# CHECK-ERROR-LABEL: vmerge.vxm v0, v1, x1, v0

vfmerge.vfm v0, v1, f1, v0
# CHECK-ERROR: The destination vector register group cannot be V0.
# CHECK-ERROR-LABEL: vfmerge.vfm v0, v1, f1, v0

vle8.v v0, (a0), v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vle8.v v0, (a0), v0.t

vfclass.v v0, v1, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfclass.v v0, v1, v0.t

vfsqrt.v v0, v1, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vfsqrt.v v0, v1, v0.t

vzext.vf2 v0, v1, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vzext.vf2 v0, v1, v0.t

vid.v v0, v0.t
# CHECK-ERROR: The destination vector register group cannot overlap the mask register.
# CHECK-ERROR-LABEL: vid.v v0, v0.t
