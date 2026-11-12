## Test R_RISCV_CALL relaxation for some boundary situations that need multiple
## iterations before symbols fit in a c.j immediate.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+c %s -o %t.rv32
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+c %s -o %t.rv64
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+zca %s -o %t.rv32zca
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32zca

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+zca %s -o %t.rv64zca
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64zca

        .globl main
        .type main,@function
main:
## Relaxed to c.j. This needs 2 iterations: c.j only fits after first relaxing
## to jal
        tail f
        .space 2042
        .size main, .-main

        .globl f
        .type f,@function
f:
## Relaxed to c.j in the same way as above.
        tail main
        .size f, .-f

        .globl g
        .type g,@function
g:
## Relaxed to c.j. This needs 3 iterations: c.j only fits after first relaxing
## both itself and the call to g to jal, and then relaxing the call to g to c.j
        tail h
        tail g
        .space 2040
        .size g, .-g

        .globl h
        .type h,@function
h:
## Relaxed to c.j in the same way as above.
        tail g
        .size h, .-h

## main: c.j f
# jitlink-check: (*{2}(main))[1:0] = 0x1
# jitlink-check: (*{2}(main))[15:13] = 0x5
# jitlink-check: decode_operand(main, 0)[11:0] = (f - main)[11:0]

## f: c.j main
# jitlink-check: (*{2}(f))[1:0] = 0x1
# jitlink-check: (*{2}(f))[15:13] = 0x5
# jitlink-check: decode_operand(f, 0)[11:0] = (main - f)[11:0]

## g: c.j h; c.j g
# jitlink-check: (*{2}(g))[1:0] = 0x1
# jitlink-check: (*{2}(g))[15:13] = 0x5
# jitlink-check: decode_operand(g, 0)[11:0] = (h - g)[11:0]
# jitlink-check: (*{2}(g+2))[1:0] = 0x1
# jitlink-check: (*{2}(g+2))[15:13] = 0x5
# jitlink-check: decode_operand(g+2, 0)[11:0] = (g - (g + 2))[11:0]

## h: c.j g
# jitlink-check: (*{2}(h))[1:0] = 0x1
# jitlink-check: (*{2}(h))[15:13] = 0x5
# jitlink-check: decode_operand(h, 0)[11:0] = (g - h)[11:0]
