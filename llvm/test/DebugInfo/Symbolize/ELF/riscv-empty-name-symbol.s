# REQUIRES: riscv-registered-target
## Ignore empty name symbols.

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYM

# SYM: 0000000000000004  0 NOTYPE LOCAL  DEFAULT [[#]] {{$}}
# SYM: 0000000000000000  0 NOTYPE GLOBAL DEFAULT [[#]] foo

## Make sure we test at an address larger than or equal to an empty name symbol.
# RUN: llvm-symbolizer --obj=%t 0 4 | FileCheck %s

# CHECK:       foo
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:
# CHECK-NEXT:  foo
# CHECK-NEXT:  ??:0:0

.globl foo
foo:
  nop
  .file 1 "/tmp" "a.s"
  .loc 1 1 0
  nop

.section .debug_line,"",@progbits
