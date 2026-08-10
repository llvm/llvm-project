@ RUN: llvm-mc < %s -triple armv7r -mattr=+hwdiv-arm -filetype=obj | llvm-objdump -d - | FileCheck %s
@ v7r implies Thumb hwdiv, but ARM hwdiv is optional
@ FIXME: Does that imply we should actually refuse to disassemble it?

.eabi_attribute Tag_CPU_arch, 10 // v7
.eabi_attribute Tag_CPU_arch_profile, 0x52 // 'R' profile

.arm
div_arm:
  udiv r0, r1, r2

@CHECK-LABEL: div_arm
@CHECK: e730f211    <unknown>

.thumb
div_thumb:
  udiv r0, r1, r2

@CHECK-LABEL: div_thumb
@CHECK: fbb1 f0f2   udiv r0, r1, r2
