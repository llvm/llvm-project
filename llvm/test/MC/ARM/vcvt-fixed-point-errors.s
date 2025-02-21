// RUN: not llvm-mc -triple=armv8a-none-eabi -mattr=+fullfp16 < %s 2>&1 | FileCheck %s

  vcvt.u16.f16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.s16.f16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.u32.f16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.s32.f16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.u16.f32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.s16.f32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.u32.f32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.s32.f32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.u16.f64 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.s16.f64 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.u32.f64 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.s32.f64 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f16.u16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f16.s16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f16.u32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f16.s32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f32.u16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f32.s16 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f32.u32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f32.s32 s0, s1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f64.u16 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f64.s16 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f64.u32 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same
  vcvt.f64.s32 d0, d1, #1
// CHECK: [[@LINE-1]]{{.*}}error: source and destination registers must be the same

