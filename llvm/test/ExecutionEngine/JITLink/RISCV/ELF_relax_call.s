# REQUIRES: asserts
# RUN: llvm-mc -triple=riscv32 -mattr=+relax -filetype=obj -o %t.rv32 %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32 2>&1 \
# RUN:     | FileCheck %s

# RUN: llvm-mc -triple=riscv64 -mattr=+relax -filetype=obj -o %t.rv64 %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64 2>&1 \
# RUN:     | FileCheck %s

        .text

## Successful relaxation: call -> jal
        .globl  main
        .type   main,@function
main:
        call f
        .size   main, .-main

        .skip  (1 << 20) - (. - main) - 2

        .globl f
        .type   f,@function
f:
        call main
        .size f, .-f

## Failed relaxation: call -> auipc, jalr
        .globl g
g:
        call h
        .size g, .-g

        .skip  (1 << 20) - (. - g) + 2

        .globl h
        .type   h,@function
h:
        call g
        .size h, .-h

## Successful relaxation: jump -> jal (not c.j as RVC is disabled)
        .globl i
        .type   i,@function
i:
        jump j, t0
        .size i, .-i

        .skip  (1 << 11) - (. - i) - 2

        .globl j
        .type   j,@function
j:
        jump i, t1
        .size j, .-j

## Successful relaxation: jump -> jal
        .globl k
        .type   k,@function
k:
        jump l, t2
        .size k, .-k

        .skip  (1 << 20) - (. - k) - 2

        .globl l
        .type   l,@function
l:
        jump k, t3
        .size l, .-l

## Failed relaxation: jump -> auipc, jalr
        .globl m
        .type   m,@function
m:
        jump n, t2
        .size m, .-m

        .skip  (1 << 20) - (. - m) + 2

        .globl n
        .type   n,@function
n:
        jump m, t3
        .size n, .-n

## Successful relaxation: call -> jal
        .globl o
        .type   o,@function
o:
        call p
        .size o, .-o

        .skip  (1 << 11) - (. - o) - 2

        .globl p
        .type   p,@function
p:
        call o
        .size p, .-p

# CHECK: Link graph before copy-and-fixup:
# CHECK: section .text:
# CHECK:   block 0x1000
# CHECK:     symbols:
# CHECK:       {{.*}} (block + 0x00000000): size: 0x00000004, linkage: strong, scope: default, live  -   main
# CHECK:       {{.*}} (block + 0x000ffffa): size: 0x00000004, linkage: strong, scope: default, live  -   f
# CHECK:       {{.*}} (block + 0x000ffffe): size: 0x00000008, linkage: strong, scope: default, live  -   g
# CHECK:       {{.*}} (block + 0x00200000): size: 0x00000008, linkage: strong, scope: default, live  -   h
# CHECK:       {{.*}} (block + 0x00200008): size: 0x00000004, linkage: strong, scope: default, live  -   i
# CHECK:       {{.*}} (block + 0x00200802): size: 0x00000004, linkage: strong, scope: default, live  -   j
# CHECK:       {{.*}} (block + 0x00200806): size: 0x00000004, linkage: strong, scope: default, live  -   k
# CHECK:       {{.*}} (block + 0x00300800): size: 0x00000004, linkage: strong, scope: default, live  -   l
# CHECK:       {{.*}} (block + 0x00300804): size: 0x00000008, linkage: strong, scope: default, live  -   m
# CHECK:       {{.*}} (block + 0x00400806): size: 0x00000008, linkage: strong, scope: default, live  -   n
# CHECK:       {{.*}} (block + 0x0040080e): size: 0x00000004, linkage: strong, scope: default, live  -   o
# CHECK:       {{.*}} (block + 0x00401008): size: 0x00000004, linkage: strong, scope: default, live  -   p
# CHECK:     edges:
# CHECK:       {{.*}} (block + 0x00000000), addend = +0x00000000, kind = R_RISCV_JAL, target = f
# CHECK:       {{.*}} (block + 0x000ffffa), addend = +0x00000000, kind = R_RISCV_JAL, target = main
# CHECK:       {{.*}} (block + 0x000ffffe), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = h
# CHECK:       {{.*}} (block + 0x00200000), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = g
# CHECK:       {{.*}} (block + 0x00200008), addend = +0x00000000, kind = R_RISCV_JAL, target = j
# CHECK:       {{.*}} (block + 0x00200802), addend = +0x00000000, kind = R_RISCV_JAL, target = i
# CHECK:       {{.*}} (block + 0x00200806), addend = +0x00000000, kind = R_RISCV_JAL, target = l
# CHECK:       {{.*}} (block + 0x00300800), addend = +0x00000000, kind = R_RISCV_JAL, target = k
# CHECK:       {{.*}} (block + 0x00300804), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = n
# CHECK:       {{.*}} (block + 0x00400806), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = m
# CHECK:       {{.*}} (block + 0x0040080e), addend = +0x00000000, kind = R_RISCV_JAL, target = p
# CHECK:       {{.*}} (block + 0x00401008), addend = +0x00000000, kind = R_RISCV_JAL, target = o

## main: jal f
# jitlink-check: (*{4}(main))[11:0] = 0xef
# jitlink-check: decode_operand(main, 1) = (f - main)

## f: jal main
# jitlink-check: (*{4}(f))[11:0] = 0xef
# jitlink-check: decode_operand(f, 1) = (main - f)

## g:
## - auipc ra, %pcrel_hi(h)
# jitlink-check: (*{4}(g))[11:0] = 0x97
# jitlink-check: decode_operand(g, 1) = (h - g + 0x800)[31:12]
## - jalr ra, %pcrel_lo(g)
# jitlink-check: (*{4}(g+4))[19:0] = 0x80e7
# jitlink-check: decode_operand(g+4, 2)[11:0] = (h - g)[11:0]

## h:
## - auipc ra, %pcrel_hi(g)
# jitlink-check: (*{4}(h))[11:0] = 0x97
# jitlink-check: decode_operand(h, 1) = (g - h + 0x800)[31:12]
## - jalr ra, %pcrel_lo(h)
# jitlink-check: (*{4}(h+4))[19:0] = 0x80e7
# jitlink-check: decode_operand(h+4, 2)[11:0] = (g - h)[11:0]

## i: jal x0, j
# jitlink-check: (*{4}(i))[11:0] = 0x6f
# jitlink-check: decode_operand(i, 1)[11:0] = (j - i)[11:0]

## j: jal x0, i
# jitlink-check: (*{4}(j))[11:0] = 0x6f
# jitlink-check: decode_operand(j, 1)[11:0] = (i - j)[11:0]

## k: jal x0, l
# jitlink-check: (*{4}(k))[11:0] = 0x6f
# jitlink-check: decode_operand(k, 1) = (l - k)

## l: jal x0, k
# jitlink-check: (*{4}(l))[11:0] = 0x6f
# jitlink-check: decode_operand(l, 1) = (k - l)

## m:
## - auipc t2, %pcrel_hi(n)
# jitlink-check: (*{4}(m))[11:0] = 0x397
# jitlink-check: decode_operand(m, 1) = (n - m + 0x800)[31:12]
## - jalr t2, %pcrel_lo(m)
# jitlink-check: (*{4}(m+4))[19:0] = 0x38067
# jitlink-check: decode_operand(m+4, 2)[11:0] = (n - m)[11:0]

## n:
## - auipc t3, %pcrel_hi(m)
# jitlink-check: (*{4}(n))[11:0] = 0xe17
# jitlink-check: decode_operand(n, 1) = (m - n + 0x800)[31:12]
## - jalr t3, %pcrel_lo(n)
# jitlink-check: (*{4}(n+4))[19:0] = 0xe0067
# jitlink-check: decode_operand(n+4, 2)[11:0] = (m - n)[11:0]

## o: jal p
# jitlink-check: (*{4}(o))[11:0] = 0xef
# jitlink-check: decode_operand(o, 1) = (p - o)

## p: jal o
# jitlink-check: (*{4}(p))[11:0] = 0xef
# jitlink-check: decode_operand(p, 1) = (o - p)
