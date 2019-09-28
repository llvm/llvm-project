@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 4
@CHECK:   .eabi_attribute 6, 4
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v5TE

.eabi_attribute  Tag_FP_arch, 4
@CHECK:   .eabi_attribute 10, 4
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: VFPv3-D16

.eabi_attribute  Tag_Advanced_SIMD_arch, 4
@CHECK:   .eabi_attribute 12, 4
@CHECK-OBJ: Tag: 12
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: Advanced_SIMD_arch
@CHECK-OBJ-NEXT: Description: ARMv8.1-a NEON

.eabi_attribute  Tag_PCS_config, 4
@CHECK:   .eabi_attribute 13, 4
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: Palm OS 2004

.eabi_attribute  Tag_ABI_PCS_wchar_t, 4
@CHECK:   .eabi_attribute 18, 4
@CHECK-OBJ: Tag: 18
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: ABI_PCS_wchar_t
@CHECK-OBJ-NEXT: Description: 4-byte

.eabi_attribute  Tag_ABI_align_needed, 4
@CHECK:   .eabi_attribute 24, 4
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: 8-byte alignment, 16-byte extended alignment

.eabi_attribute  Tag_ABI_align_preserved, 4
@CHECK:   .eabi_attribute 25, 4
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: 8-byte stack alignment, 16-byte data alignment

.eabi_attribute  Tag_ABI_optimization_goals, 4
@CHECK:   .eabi_attribute 30, 4
@CHECK-OBJ: Tag: 30
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: ABI_optimization_goals
@CHECK-OBJ-NEXT: Description: Aggressive Size

.eabi_attribute  Tag_ABI_FP_optimization_goals, 4
@CHECK:   .eabi_attribute 31, 4
@CHECK-OBJ: Tag: 31
@CHECK-OBJ-NEXT: Value: 4
@CHECK-OBJ-NEXT: TagName: ABI_FP_optimization_goals
@CHECK-OBJ-NEXT: Description: Aggressive Size

