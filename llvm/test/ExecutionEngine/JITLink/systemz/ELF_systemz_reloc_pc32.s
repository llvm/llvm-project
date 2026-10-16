# REQUIRES: system-linux
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -defsym OFFSET=0x80000000 -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -abs OFFSET=0x80000000 -check=%s %t.o
#
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -defsym OFFSET=0xFFFFFFFF -filetype=obj -o %t.o %s
# RUN: not llvm-jitlink -noexec -abs OFFSET=0xFFFFFFFF %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=CHECK-ERROR %s
#
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -defsym OFFSET=0x80000001 -filetype=obj -o %t.o %s
# RUN: not llvm-jitlink -noexec -abs OFFSET=0x80000001 %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=CHECK-ERROR %s
#
# jitlink-check: *{4}test_pc32 = OFFSET
# jitlink-check: *{4}test_pc32dbl = OFFSET

# CHECK-ERROR:  {{.*}} is out of range of Delta32 fixup

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br    %r14
        .size   main, .-main

	.globl test_pc32
test_pc32:
  	.reloc test_pc32, R_390_PC32, .-OFFSET
	.space 4 
  	.size test_pc32, .-test_pc32 

	.globl test_pc32dbl
test_pc32dbl:
  	.reloc test_pc32dbl, R_390_PC32DBL, .-(OFFSET + OFFSET)
	.space 4 
  	.size test_pc32dbl, .-test_pc32dbl 

