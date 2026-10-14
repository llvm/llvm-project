# REQUIRES: asserts
## Test that sections with many .prefalign fragments converge quickly.
## PrefAlign fragments see fresh offsets and converge in 1 iteration.

# RUN: llvm-mc -filetype=obj -triple x86_64 --stats %s -o %t 2>&1 | FileCheck %s --check-prefix=STATS
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# STATS: 2 assembler - Number of assembler layout and relaxation steps

# CHECK:       8: int3
# CHECK-NEXT:  9: int3
# CHECK-NEXT:  a: int3
# CHECK-NEXT:  b: int3
# CHECK-NEXT:  c: int3
# CHECK-NEXT:  d: nopl
# CHECK-NEXT: 10: int3
# CHECK:      15: nopl
# CHECK-NEXT: 18: int3
# CHECK:      1d: nopl
# CHECK-NEXT: 20: int3
# CHECK:      25: nopl
# CHECK-NEXT: 28: int3
# CHECK:      2d: nopl
# CHECK-NEXT: 30: int3
# CHECK:      35: nopl
# CHECK-NEXT: 38: int3
# CHECK:      3d: nopl
# CHECK-NEXT: 40: int3
# CHECK:      45: nopl
# CHECK-NEXT: 48: int3
# CHECK:      4d: nopl
# CHECK-NEXT: 50: int3
# CHECK-NEXT: 51: int3
# CHECK-NEXT: 52: int3
# CHECK-NEXT: 53: int3
# CHECK-NEXT: 54: int3

.section .text,"ax",@progbits
.byte 0

.rept 10
.prefalign 4, .Lend\+, nop
.rept 5
int3
.endr
.Lend\+:
.endr
