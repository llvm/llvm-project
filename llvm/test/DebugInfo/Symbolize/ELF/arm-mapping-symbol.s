# REQUIRES: arm-registered-target
## Ignore ARM mapping symbols (with a prefix of $a, $d or $t).

# RUN: llvm-mc -filetype=obj -triple=armv7-none-linux %s -o %t
# RUN: llvm-symbolizer --obj=%t 4 8 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=thumbv7-none-linux %s -o %tthumb
# RUN: llvm-symbolizer --obj=%tthumb 4 8 | FileCheck %s

# CHECK:       foo
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:
# CHECK-NEXT:  foo
# CHECK-NEXT:  ??:0:0

.globl foo
foo:
  .word 32
  nop
  .word 32
