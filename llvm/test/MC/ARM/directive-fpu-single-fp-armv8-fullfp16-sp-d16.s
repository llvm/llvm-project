@ Check a single .fpu directive.

@ RUN: llvm-mc -triple arm-eabi -filetype obj %s \
@ RUN:   | llvm-readobj --arch-specific - \
@ RUN:   | FileCheck %s -check-prefix CHECK-ATTR

  .fpu fp-armv8-fullfp16-sp-d16

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: FP_arch
@ CHECK-ATTR:     Description: ARMv8-a FP-D16
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

