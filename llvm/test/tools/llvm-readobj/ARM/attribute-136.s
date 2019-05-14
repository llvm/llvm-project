@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arm-attributes - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_compatibility, 136, "Foo Corp"
@CHECK:   .eabi_attribute 32, 136
@CHECK-OBJ: Tag: 32
@CHECK-OBJ-NEXT: Value: 136, Foo Corp
@CHECK-OBJ-NEXT: TagName: compatibility
@CHECK-OBJ-NEXT: Description: AEABI Non-Conformant

