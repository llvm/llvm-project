@ RUN: llvm-mc -triple arm -filetype obj -o - %s | \
@ RUN: llvm-readobj -A - 2>&1 | \
@ RUN: FileCheck %s --check-prefix=CHECK-WARNING

.eabi_attribute Tag_also_compatible_with, "\074\001"
@ CHECK-WARNING: 60 is not a valid tag number
