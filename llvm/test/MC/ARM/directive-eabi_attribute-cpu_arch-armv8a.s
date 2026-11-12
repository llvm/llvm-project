@ RUN: llvm-mc -triple arm -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple arm -filetype obj -o - %s | llvm-readobj -A - | \
@ RUN: FileCheck %s --check-prefix=CHECK-OBJ

.eabi_attribute Tag_CPU_arch, 14
@ CHECK:          .eabi_attribute 6, 14 @ Tag_CPU_arch
@ CHECK-OBJ:      Attribute
@ CHECK-OBJ: Tag: 6
@ CHECK-OBJ-NEXT: Value: 14
@ CHECK-OBJ-NEXT: TagName: CPU_arch
@ CHECK-OBJ-NEXT: Description: ARM v8-A
