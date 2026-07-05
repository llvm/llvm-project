@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 12
@CHECK:   .eabi_attribute 6, 12
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 12
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v6S-M

.eabi_attribute  Tag_ABI_align_needed, 12
@CHECK:   .eabi_attribute 24, 12
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 12
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: 8-byte alignment, 4096-byte extended alignment

.eabi_attribute  Tag_ABI_align_preserved, 12
@CHECK:   .eabi_attribute 25, 12
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 12
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: 8-byte stack alignment, 4096-byte data alignment

