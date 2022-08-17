@ RUN: llvm-mc -triple arm -filetype obj -o - %s | \
@ RUN: llvm-readobj -A - | \
@ RUN: FileCheck %s

.eabi_attribute Tag_also_compatible_with, "\015\001"
@ CHECK: Attribute
@ CHECK: Tag: 65
@ CHECK: TagName: also_compatible_with
@ CHECK: Value: \015\001
@ CHECK: Description: Tag_PCS_config = 1
