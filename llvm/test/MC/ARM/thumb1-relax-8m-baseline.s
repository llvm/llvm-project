@ RUN: not llvm-mc -triple thumbv6m-none-eabi -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-V6M
@ RUN: llvm-mc -triple thumbv8m.base-none-eabi -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=CHECK-V8MBASE --allow-empty

@ CHECK-V8MBASE-NOT: out of range pc-relative fixup value
@ CHECK-V6M: out of range pc-relative fixup value
  b Lfar2

  .space 2050
Lfar2:
  .word 42
