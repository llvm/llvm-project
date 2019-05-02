@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arm-attributes - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_compatibility, 15, "Longer Corporation NaMe"
@CHECK:   .eabi_attribute 32, 15
@CHECK-OBJ: Tag: 32
@CHECK-OBJ-NEXT: Value: 15, Longer Corporation NaMe
@CHECK-OBJ-NEXT: TagName: compatibility
@CHECK-OBJ-NEXT: Description: AEABI Non-Conformant

