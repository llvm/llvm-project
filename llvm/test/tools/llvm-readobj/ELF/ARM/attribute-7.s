@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 7
@CHECK:   .eabi_attribute 6, 7
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 7
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v6KZ

.eabi_attribute  Tag_FP_arch, 7
@CHECK:   .eabi_attribute 10, 7
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 7
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: ARMv8-a FP

.eabi_attribute  Tag_PCS_config, 7
@CHECK:   .eabi_attribute 13, 7
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 7
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: Reserved (Symbian OS)

.eabi_attribute  Tag_ABI_align_needed, 7
@CHECK:   .eabi_attribute 24, 7
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 7
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: 8-byte alignment, 128-byte extended alignment

.eabi_attribute  Tag_ABI_align_preserved, 7
@CHECK:   .eabi_attribute 25, 7
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 7
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: 8-byte stack alignment, 128-byte data alignment

