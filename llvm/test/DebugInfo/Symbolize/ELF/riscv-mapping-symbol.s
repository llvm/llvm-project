# REQUIRES: riscv-registered-target
## Ignore RISC-V mapping symbols (with a prefix of $d or $x).

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t
# RUN: llvm-symbolizer --obj=%t 0 4 0xc | FileCheck %s

# CHECK:      foo
# CHECK-NEXT: ??:0:0
# CHECK-EMPTY:
# CHECK:      foo
# CHECK-NEXT: ??:0:0
# CHECK-EMPTY:
# CHECK:      foo
# CHECK-NEXT: ??:0:0

    .global foo
foo:
    .word 32
    nop
    nop
    .word 42
