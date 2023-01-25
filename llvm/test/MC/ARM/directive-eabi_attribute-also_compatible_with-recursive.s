@ RUN: llvm-mc -triple arm -filetype obj -o - %s | \
@ RUN: llvm-readobj -A - 2>&1 | \
@ RUN: FileCheck %s --check-prefix=CHECK-WARNING

.eabi_attribute Tag_also_compatible_with, "\101\006\017"
@ CHECK-WARNING: Tag_also_compatible_with cannot be recursively defined
