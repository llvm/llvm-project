@ RUN: llvm-mc < %s -triple armv7r -mattr=+hwdiv-arm -filetype=obj | llvm-objdump -triple=thumb -d - | FileCheck %s
@ RUN: llvm-mc < %s -triple armv7r -mattr=+hwdiv-arm -filetype=obj | llvm-objdump -triple=arm -d - | FileCheck %s --check-prefix=CHECK-ARM

.eabi_attribute Tag_CPU_arch, 10 // v7
.eabi_attribute Tag_CPU_arch_profile, 0x52 // 'R' profile

.arm
div_arm:
  udiv r0, r1, r2

@CHECK-LABEL: div_arm
@CHECK-NOT: udiv r0, r1, r2
@CHECK-ARM-NOT: udiv r0, r1, r2

.thumb
div_thumb:
  udiv r0, r1, r2

@CHECK-LABEL: div_thumb
@CHECK: b1 fb f2 f0 udiv r0, r1, r2
