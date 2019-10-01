@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arm-attributes - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch_profile, 'S'
@CHECK:   .eabi_attribute 7, 83
@CHECK-OBJ: Tag: 7
@CHECK-OBJ-NEXT: Value: 83
@CHECK-OBJ-NEXT: TagName: CPU_arch_profile
@CHECK-OBJ-NEXT: Description: Classic

