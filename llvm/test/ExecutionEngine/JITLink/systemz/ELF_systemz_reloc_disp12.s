# REQUIRES: system-linux

# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
#
# RUN: llvm-jitlink -noexec -abs DISP=0xFFF -check=%s %t.o

# RUN: not llvm-jitlink -noexec -abs  DISP=0x1000 %t.o 2>&1 | \
# RUN:  FileCheck -check-prefix=CHECK-ERROR %s
#
# Check success and failure cases of R_390_12 handling.

# CHECK-ERROR: relocation target {{.*}} (DISP) is out of range of
# CHECK-ERROR: Pointer12 fixup

# jitlink-check: decode_operand(main, 2) = DISP
        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
    	.reloc .+2, R_390_12, DISP 
    	l %r6, 0(%r7,%r8)
        br    %r14
.Lfunc_end0:
        .size   main, .Lfunc_end0-main

