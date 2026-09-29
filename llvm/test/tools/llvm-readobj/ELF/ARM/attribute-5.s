@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 5
@CHECK:   .eabi_attribute 6, 5
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 5
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v5TEJ

.eabi_attribute  Tag_FP_arch, 5
@CHECK:   .eabi_attribute 10, 5
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 5
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: VFPv4

.eabi_attribute  Tag_PCS_config, 5
@CHECK:   .eabi_attribute 13, 5
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 5
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: Reserved (Palm OS)

.eabi_attribute  Tag_ABI_align_needed, 5
@CHECK:   .eabi_attribute 24, 5
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 5
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: 8-byte alignment, 32-byte extended alignment

.eabi_attribute  Tag_ABI_align_preserved, 5
@CHECK:   .eabi_attribute 25, 5
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 5
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: 8-byte stack alignment, 32-byte data alignment

.eabi_attribute  Tag_ABI_optimization_goals, 5
@CHECK:   .eabi_attribute 30, 5
@CHECK-OBJ: Tag: 30
@CHECK-OBJ-NEXT: Value: 5
@CHECK-OBJ-NEXT: TagName: ABI_optimization_goals
@CHECK-OBJ-NEXT: Description: Debugging

.eabi_attribute  Tag_ABI_FP_optimization_goals, 5
@CHECK:   .eabi_attribute 31, 5
@CHECK-OBJ: Tag: 31
@CHECK-OBJ-NEXT: Value: 5
@CHECK-OBJ-NEXT: TagName: ABI_FP_optimization_goals
@CHECK-OBJ-NEXT: Description: Accuracy

