@ RUN: llvm-mc < %s -triple armv7a -mattr=+vfp3,+neon,+fp16,+hwdiv-arm,+hwdiv -filetype=obj | llvm-objdump --triple=arm -d - | FileCheck %s
@ RUN: llvm-mc < %s -triple armv7a -mattr=+vfp3,+neon,+fp16,+hwdiv-arm,+hwdiv -filetype=obj | llvm-objdump --triple=thumb -d - | FileCheck %s --check-prefix=CHECK-THUMB

.eabi_attribute Tag_FP_arch, 0 // disallow vfp

vfp2:
  vmla.f32 s0, s1, s2

@CHECK-LABEL: vfp2
@CHECK-NOT: 81 0a 00 ee vmla.f32 s0, s1, s2
@CHECK: unknown

vfp3:
  vmov.f32 s0, #0.5

@CHECK-LABEL: vfp3
@CHECK-NOT: 00 0a b6 ee vmov.f32 s0, #5.000000e-01

neon:
  vmla.f32 d0, d1, d2

@CHECK-LABEL: neon
@CHECK-NOT: 12 0d 01 f2 vmla.f32 d0, d1, d2
@CHECK: unknown

fp16:
  vcvt.f32.f16 q0, d2

@CHECK-LABEL: fp16
@CHECK-NOT: 02 07 b6 f3  vcvt.f32.f16 q0, d2

div_arm:
  udiv r0, r1, r2

@CHECK-LABEL: div_arm
@CHECK-NOT: 11 f2 30 e7 udiv r0, r1, r2
@CHECK: unknown

.thumb
div_thumb:
  udiv r0, r1, r2

@CHECK-LABEL: div_thumb
@CHECK-THUMB-NOT: b1 fb f2 f0 udiv r0, r1, r2
