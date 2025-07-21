# REQUIRES: asserts
# RUN: llvm-mc -triple=riscv32 -mattr=+relax,+c -filetype=obj -o %t.rv32 %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32 2>&1 \
# RUN:     | FileCheck %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s -check-name=jitlink-check-rv32 %t.rv32 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# RUN: llvm-mc -triple=riscv64 -mattr=+relax,+c -filetype=obj -o %t.rv64 %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64 2>&1 \
# RUN:     | FileCheck %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s -check-name=jitlink-check-rv64 %t.rv64 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV64 %s

# RUN: llvm-mc -triple=riscv32 -mattr=+relax,+zca -filetype=obj -o %t.rv32zca %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv32zca 2>&1 \
# RUN:     | FileCheck %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s -check-name=jitlink-check-rv32 %t.rv32zca 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# RUN: llvm-mc -triple=riscv64 -mattr=+relax,+c -filetype=obj -o %t.rv64 %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s %t.rv64 2>&1 \
# RUN:     | FileCheck %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x1000 -slab-page-size 4096 \
# RUN:     -check %s -check-name=jitlink-check-rv64 %t.rv64 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV64 %s

        .text

## Successful relaxation: call -> jal
        .globl  main
        .type   main,@function
main:
        call f # rv64+c: jal (size 4)
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

## Successful relaxation: jump -> c.j
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

## RV32: Successful relaxation: call -> c.jal
## RV64: Successful relaxation: call -> jal
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

# CHECK:      Link graph "{{.*}}" before copy-and-fixup:
# CHECK:      section .text:
# CHECK:        block 0x1000
# CHECK:          symbols:
# CHECK:            {{.*}} (block + 0x00000000): size: 0x00000004, linkage: strong, scope: default, live  -   main
# CHECK:            {{.*}} (block + 0x000ffffa): size: 0x00000004, linkage: strong, scope: default, live  -   f
# CHECK:            {{.*}} (block + 0x000ffffe): size: 0x00000008, linkage: strong, scope: default, live  -   g
# CHECK:            {{.*}} (block + 0x00200000): size: 0x00000008, linkage: strong, scope: default, live  -   h
# CHECK:            {{.*}} (block + 0x00200008): size: 0x00000002, linkage: strong, scope: default, live  -   i
# CHECK:            {{.*}} (block + 0x00200800): size: 0x00000002, linkage: strong, scope: default, live  -   j
# CHECK:            {{.*}} (block + 0x00200802): size: 0x00000004, linkage: strong, scope: default, live  -   k
# CHECK:            {{.*}} (block + 0x003007fc): size: 0x00000004, linkage: strong, scope: default, live  -   l
# CHECK:            {{.*}} (block + 0x00300800): size: 0x00000008, linkage: strong, scope: default, live  -   m
# CHECK:            {{.*}} (block + 0x00400802): size: 0x00000008, linkage: strong, scope: default, live  -   n
# CHECK-RV32:       {{.*}} (block + 0x0040080a): size: 0x00000002, linkage: strong, scope: default, live  -   o
# CHECK-RV64:       {{.*}} (block + 0x0040080a): size: 0x00000004, linkage: strong, scope: default, live  -   o
# CHECK-RV32:       {{.*}} (block + 0x00401002): size: 0x00000002, linkage: strong, scope: default, live  -   p
# CHECK-RV64:       {{.*}} (block + 0x00401004): size: 0x00000004, linkage: strong, scope: default, live  -   p
# CHECK:          edges:
# CHECK:            {{.*}} (block + 0x00000000), addend = +0x00000000, kind = R_RISCV_JAL, target = f
# CHECK:            {{.*}} (block + 0x000ffffa), addend = +0x00000000, kind = R_RISCV_JAL, target = main
# CHECK:            {{.*}} (block + 0x000ffffe), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = h
# CHECK:            {{.*}} (block + 0x00200000), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = g
# CHECK:            {{.*}} (block + 0x00200008), addend = +0x00000000, kind = R_RISCV_RVC_JUMP, target = j
# CHECK:            {{.*}} (block + 0x00200800), addend = +0x00000000, kind = R_RISCV_RVC_JUMP, target = i
# CHECK:            {{.*}} (block + 0x00200802), addend = +0x00000000, kind = R_RISCV_JAL, target = l
# CHECK:            {{.*}} (block + 0x003007fc), addend = +0x00000000, kind = R_RISCV_JAL, target = k
# CHECK:            {{.*}} (block + 0x00300800), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = n
# CHECK:            {{.*}} (block + 0x00400802), addend = +0x00000000, kind = R_RISCV_CALL_PLT, target = m
# CHECK-RV32:       {{.*}} (block + 0x0040080a), addend = +0x00000000, kind = R_RISCV_RVC_JUMP, target = p
# CHECK-RV64:       {{.*}} (block + 0x0040080a), addend = +0x00000000, kind = R_RISCV_JAL, target = p
# CHECK-RV32:       {{.*}} (block + 0x00401002), addend = +0x00000000, kind = R_RISCV_RVC_JUMP, target = o
# CHECK-RV64:       {{.*}} (block + 0x00401004), addend = +0x00000000, kind = R_RISCV_JAL, target = o

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

## i: c.j j
# jitlink-check: (*{2}(i))[1:0] = 0x1
# jitlink-check: (*{2}(i))[15:13] = 0x5
# jitlink-check: decode_operand(i, 0)[11:0] = (j - i)[11:0]

## j: c.j i
# jitlink-check: (*{2}(j))[1:0] = 0x1
# jitlink-check: (*{2}(j))[15:13] = 0x5
# jitlink-check: decode_operand(j, 0)[11:0] = (i - j)[11:0]

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

## RV32: o: c.jal p
# jitlink-check-rv32: (*{2}(o))[1:0] = 0x1
# jitlink-check-rv32: (*{2}(o))[15:13] = 0x1
# jitlink-check-rv32: decode_operand(o, 0) = (p - o)

## RV64: o: jal p
# jitlink-check-rv64: (*{4}(o))[11:0] = 0xef
# jitlink-check-rv64: decode_operand(o, 1) = (p - o)

## RV32: p: c.jal o
# jitlink-check-rv32: (*{2}(p))[1:0] = 0x1
# jitlink-check-rv32: (*{2}(p))[15:13] = 0x1
# jitlink-check-rv32: decode_operand(p, 0) = (o - p)

## RV64: p: jal o
# jitlink-check-rv64: (*{4}(p))[11:0] = 0xef
# jitlink-check-rv64: decode_operand(p, 1) = (o - p)
