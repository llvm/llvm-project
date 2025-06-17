# REQUIRES: system-linux
# RUN: llvm-mc -triple=systemz-unknown-linux -mcpu=z16 -position-independent \
# RUN:         -defsym OFF12=0xffe -defsym OFF16=4 -defsym OFF24=6 \
# RUN:         -defsym OFF32=6 -filetype=obj -o %t.o %s
#
# RUN: llvm-jitlink -noexec -abs OFF12=0xffe -abs OFF16=4 -abs OFF24=6 \
# RUN:                      -abs OFF32=6 -check=%s %t.o
#
# RUN: llvm-mc -triple=systemz-unknown-linux -mcpu=z16 -position-independent \
# RUN:         -defsym OFF12=6 -defsym OFF16=0xfffe -defsym OFF24=6 \
# RUN:         -defsym OFF32=6 -filetype=obj -o %t.o %s
#
# RUN: llvm-jitlink -noexec -abs OFF12=6 -abs OFF16=0xfffe -abs OFF24=6 \
# RUN:                      -abs OFF32=6 -check=%s %t.o
#
# RUN: llvm-mc -triple=systemz-unknown-linux -mcpu=z16 -position-independent \
# RUN:         -defsym OFF12=6 -defsym OFF16=4 -defsym OFF24=0xfffffe \
# RUN:         -defsym OFF32=6 -filetype=obj -o %t.o %s
#
# RUN: llvm-jitlink -noexec -abs OFF12=6 -abs OFF16=4 -abs OFF24=0xfffffe \
# RUN:                      -abs OFF32=6 -check=%s %t.o
#
# RUN: llvm-mc -triple=systemz-unknown-linux -mcpu=z16 -position-independent \
# RUN:         -defsym OFF12=6 -defsym OFF16=4 -defsym OFF24=6 \
# RUN:         -defsym OFF32=0xffffffc8 -filetype=obj -o %t.o %s
#
# RUN: llvm-jitlink -noexec -abs OFF12=6 -abs OFF16=4 -abs OFF24=6 \
# RUN:                      -abs OFF32=0xffffffc8 -check=%s %t.o

# Check R_390_PC*dbl relocations.

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br   %r14
        .size main, .-main

# R_390_PC16DBL
# jitlink-check: *{2}(test_pc16dbl + 2) = (OFF16 >> 1)
        .globl  test_pc16dbl
        .p2align 3 
test_pc16dbl:
        je   .Lpc16dbl
	.space OFF16 - 4
.Lpc16dbl:
        jne  test_pc16dbl 
        .size test_pc16dbl,.-test_pc16dbl

# R_390_PC32DBL
# jitlink-check: *{4}(test_pc32dbl + 2) = (OFF32 >> 1)
        .globl  test_pc32dbl
        .p2align 3 
test_pc32dbl:
        jge   .Lpc32dbl
	.space OFF32 - 6 
.Lpc32dbl:
        jgne  test_pc32dbl
        .size test_pc32dbl,.-test_pc32dbl

# R_390_PC12DBL
# jitlink-check: ((*{2} (test_pc12dbl + 1)) & 0x0fff) = (OFF12 >> 1)
        .globl  test_pc12dbl
        .p2align 4 
test_pc12dbl:
        bprp  0, .Lpc12dbl, 0 
        .space OFF12 - 6
.Lpc12dbl:
        bprp  0, test_pc12dbl, 0 
        .size test_pc12dbl,.-test_pc12dbl

# R_390_PC24DBL
# jitlink-check: ((*{4} (test_pc24dbl + 2)) & 0x0ffffff) = (OFF24 >> 1)
        .globl  test_pc24dbl
        .p2align 4 
test_pc24dbl:
        bprp  0, 0, .Lpc24dbl
        .space OFF24 - 6
.Lpc24dbl:
        bprp  0, 0, test_pc24dbl
        .size test_pc24dbl,.-test_pc24dbl

