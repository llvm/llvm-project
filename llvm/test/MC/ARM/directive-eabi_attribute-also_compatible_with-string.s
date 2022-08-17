@ RUN: llvm-mc -triple arm -filetype obj -o - %s | \
@ RUN: llvm-readobj -A - | \
@ RUN: FileCheck %s

.eabi_attribute Tag_also_compatible_with, "\005Cortex-A7"
@ CHECK: Attribute
@ CHECK: Tag: 65
@ CHECK: TagName: also_compatible_with
@ CHECK: Value: \005Cortex-A7
@ CHECK: Description: Tag_CPU_name = Cortex-A7
