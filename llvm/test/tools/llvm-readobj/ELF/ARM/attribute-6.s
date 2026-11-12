@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 6
@CHECK:   .eabi_attribute 6, 6
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 6
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v6

.eabi_attribute  Tag_FP_arch, 6
@CHECK:   .eabi_attribute 10, 6
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 6
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: VFPv4-D16

.eabi_attribute  Tag_PCS_config, 6
@CHECK:   .eabi_attribute 13, 6
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 6
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: Symbian OS 2004

.eabi_attribute  Tag_ABI_align_needed, 6
@CHECK:   .eabi_attribute 24, 6
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 6
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: 8-byte alignment, 64-byte extended alignment

.eabi_attribute  Tag_ABI_align_preserved, 6
@CHECK:   .eabi_attribute 25, 6
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 6
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: 8-byte stack alignment, 64-byte data alignment

.eabi_attribute  Tag_ABI_optimization_goals, 6
@CHECK:   .eabi_attribute 30, 6
@CHECK-OBJ: Tag: 30
@CHECK-OBJ-NEXT: Value: 6
@CHECK-OBJ-NEXT: TagName: ABI_optimization_goals
@CHECK-OBJ-NEXT: Description: Best Debugging

.eabi_attribute  Tag_ABI_FP_optimization_goals, 6
@CHECK:   .eabi_attribute 31, 6
@CHECK-OBJ: Tag: 31
@CHECK-OBJ-NEXT: Value: 6
@CHECK-OBJ-NEXT: TagName: ABI_FP_optimization_goals
@CHECK-OBJ-NEXT: Description: Best Accuracy

