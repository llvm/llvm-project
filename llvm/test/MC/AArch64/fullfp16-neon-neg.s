// RUN: not llvm-mc -triple=aarch64 -mattr=+neon,-fullfp16 -show-encoding < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=aarch64 -mattr=-neon,+fullfp16 -show-encoding < %s 2>&1 | FileCheck %s


// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fabs.4h     v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fneg.4h     v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecpe.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinta.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintx.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinti.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintm.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintn.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintp.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintz.4h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrte.4h  v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fsqrt.4h    v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fabs.8h     v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fneg.8h     v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecpe.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinta.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintx.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinti.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintm.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintn.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintp.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintz.8h   v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrte.8h  v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fsqrt.8h    v0, v0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmla v0.4h, v1.4h, v2.h[2]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmla v3.8h, v8.8h, v2.h[1]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmls v0.4h, v1.4h, v2.h[2]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmls v3.8h, v8.8h, v2.h[1]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmul v0.4h, v1.4h, v2.h[2]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmul v0.8h, v1.8h, v2.h[2]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmulx v0.4h, v1.4h, v2.h[2]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmulx v0.8h, v1.8h, v2.h[2]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fabd v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxnmv h0, v1.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminnmv h0, v1.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxv h0, v1.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminv h0, v1.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  faddp v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  faddp v0.8h, v1.8h, v2.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fadd v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fadd v0.8h, v1.8h, v2.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fsub v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fsub v0.8h, v1.8h, v2.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq v0.4h, v31.4h, v16.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq v4.8h, v7.8h, v15.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge v3.4h, v8.4h, v12.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge v31.8h, v29.8h, v28.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle v3.4h,  v12.4h, v8.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle v31.8h, v28.8h, v29.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt v0.4h, v31.4h, v16.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt v4.8h, v7.8h, v15.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt v0.4h, v16.4h, v31.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt v4.8h, v15.8h, v7.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq v0.4h, v31.4h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq v4.8h, v7.8h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq v0.4h, v31.4h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq v4.8h, v7.8h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge v3.4h, v8.4h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge v31.8h, v29.8h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge v3.4h, v8.4h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge v31.8h, v29.8h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt v0.4h, v31.4h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt v4.8h, v7.8h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt v0.4h, v31.4h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt v4.8h, v7.8h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle v3.4h, v20.4h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle v1.8h, v8.8h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle v3.4h, v20.4h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle v1.8h, v8.8h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt v16.4h, v2.4h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt v15.8h, v4.8h, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt v16.4h, v2.4h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt v15.8h, v4.8h, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facge v0.4h, v31.4h, v16.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facge v4.8h, v7.8h, v15.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facle v0.4h, v16.4h, v31.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facle v4.8h, v15.8h, v7.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facgt v3.4h, v8.4h, v12.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facgt v31.8h, v29.8h, v28.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  faclt v3.4h,  v12.4h, v8.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  faclt v31.8h, v28.8h, v29.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrts v0.4h, v31.4h, v16.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrts v4.8h, v7.8h, v15.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecps v3.4h, v8.4h, v12.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecps v31.8h, v29.8h, v28.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxp v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxp v31.8h, v15.8h, v16.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminp v10.4h, v15.4h, v22.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminp v3.8h, v5.8h, v6.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxnmp v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxnmp v31.8h, v15.8h, v16.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminnmp v10.4h, v15.4h, v22.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminnmp v3.8h, v5.8h, v6.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmax v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmax v0.8h, v1.8h, v2.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmin v10.4h, v15.4h, v22.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmin v10.8h, v15.8h, v22.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxnm v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmaxnm v0.8h, v1.8h, v2.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminnm v10.4h, v15.4h, v22.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fminnm v10.8h, v15.8h, v22.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmla v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmla v0.8h, v1.8h, v2.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmls v0.4h, v1.4h, v2.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmls v0.8h, v1.8h, v2.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fabd h29, h24, h20
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmla    h0, h1, v1.h[5]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmls    h2, h3, v4.h[5]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmul    h0, h1, v1.h[5]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmulx   h6, h2, v8.h[5]
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzs h21, h12, #1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzu h21, h12, #1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtas h12, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtau h12, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtms h22, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtmu h12, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtns h22, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtnu h12, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtps h22, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtpu h12, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzs h12, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzu h12, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq h10, h11, h12
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq h10, h11, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmeq h10, h11, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge h10, h11, h12
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge h10, h11, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmge h10, h11, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt h10, h11, h12
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt h10, h11, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmgt h10, h11, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle h10, h11, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmle h10, h11, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt h10, h11, #0.0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcmlt h10, h11, #0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facge h10, h11, h12
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  facgt h10, h11, h12
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fmulx h20, h22, h15
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecps h21, h16, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrts h21, h5, h12
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecpe h19, h14
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecpx h18, h10
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrte h22, h13
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  faddp h18, v3.2h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fabs v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fabs v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fneg v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fneg v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintn v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintn v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinta v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinta v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintp v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintp v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintm v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintm v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintx v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintx v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintz v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frintz v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinti v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frinti v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtns v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtns v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtnu v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtnu v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtps v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtps v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtpu v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtpu v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtms v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtms v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtmu v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtmu v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzs v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzs v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzu v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtzu v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtas v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtas v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtau v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fcvtau v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecpe v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frecpe v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrte v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  frsqrte v6.8h, v8.8h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fsqrt v4.4h, v0.4h
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
  fsqrt v6.8h, v8.8h

// CHECK-NOT: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires:
