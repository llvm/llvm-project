@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj -A - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_conformance, "A.long--non numeric oddity...!!"
@CHECK:   .eabi_attribute 67, "A.long--non numeric oddity...!!"
@CHECK-OBJ: Tag: 67
@CHECK-OBJ-NEXT: TagName: conformance
@CHECK-OBJ-NEXT: Value: A.long--non numeric oddity...!!
