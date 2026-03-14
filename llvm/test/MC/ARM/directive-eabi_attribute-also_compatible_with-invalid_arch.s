@ RUN: llvm-mc -triple arm -filetype obj -o - %s | \
@ RUN: llvm-readobj -A - 2>&1 | \
@ RUN: FileCheck %s --check-prefix=CHECK-WARNING

.eabi_attribute Tag_also_compatible_with, "\006\143"
@ CHECK-WARNING: 99 is not a valid Tag_CPU_arch value
