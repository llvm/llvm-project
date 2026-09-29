# REQUIRES: asserts
## Test that an internal alignment inside a .prefalign body does not cause oscillation.

# RUN: llvm-mc -filetype=obj -triple x86_64 --stats %s -o %t 2>&1 | FileCheck %s --check-prefix=STATS
# RUN: llvm-readelf -s %t | FileCheck %s

## Converges in a small fixed number of layout passes.
# STATS: 3 assembler - Number of assembler layout and relaxation steps

# CHECK: {{0*}}20 0 NOTYPE LOCAL DEFAULT 2 c
# CHECK: {{0*}}40 0 NOTYPE LOCAL DEFAULT 2 e
# CHECK:  {{0*}}8 0 NOTYPE LOCAL DEFAULT 3 c2
# CHECK:  {{0*}}c 0 NOTYPE LOCAL DEFAULT 3 e2

.file 0 "." "t.c"
.text
.byte 0
.loc 0 1 0
.prefalign 5, .Lend0, nop
c:
.loc 0 2 0
.p2align 5
jmp c
.Lend0:

.prefalign 5, .Lend1, nop
e:
.loc 0 3 0
.p2align 5
jmp e
.Lend1:

.section .text.smaller,"ax",@progbits
.byte 0
.loc 0 4 0
.prefalign 5, .Lend2, nop
c2:
.loc 0 5 0
.p2align 2
jmp c2
.Lend2:

.prefalign 5, .Lend3, nop
e2:
.loc 0 6 0
.p2align 2
jmp e2
.Lend3:
