@ RUN: llvm-mc < %s -triple armv7m -mattr=+vfp4 -filetype=obj | llvm-objdump -d - | FileCheck %s

.eabi_attribute Tag_CPU_arch, 10 // v7
.eabi_attribute Tag_CPU_arch_profile, 0x4D // 'M' profile
.eabi_attribute Tag_FP_arch, 5 // VFP4

.thumb
vfp2:
  vmla.f32 s0, s1, s2

@CHECK-LABEL: vfp2
@CHECK: ee00 0a81   vmla.f32 s0, s1, s2

.thumb
vfp4:
  vmov.f32 s0, #0.5

@CHECK-LABEL: vfp4
@CHECK: eeb6 0a00   vmov.f32 s0, #5.000000e-01

.thumb
div:
  udiv r0, r1, r2

@CHECK-LABEL: div
@CHECK: fbb1 f0f2   udiv r0, r1, r2
