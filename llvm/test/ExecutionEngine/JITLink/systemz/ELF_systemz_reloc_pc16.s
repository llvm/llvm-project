# REQUIRES: system-linux
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -defsym OFFSET=0x8000 -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -abs OFFSET=0x8000 -check=%s %t.o
#
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -defsym OFFSET=0xFFFF -filetype=obj -o %t.o %s
# RUN: not llvm-jitlink -noexec -abs OFFSET=0xFFFF %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=CHECK-ERROR %s
#
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -defsym OFFSET=0x8001 -filetype=obj -o %t.o %s
# RUN: not llvm-jitlink -noexec -abs OFFSET=0x8001 %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=CHECK-ERROR %s
#
# jitlink-check: *{2}test_pc16 = OFFSET
# jitlink-check: *{2}test_pc16dbl = OFFSET

# CHECK-ERROR:  {{.*}} is out of range of Delta16 fixup

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br    %r14
        .size   main, .-main

	.globl test_pc16
test_pc16:
  	.reloc test_pc16, R_390_PC16, .-OFFSET
	.space 2
  	.size test_pc16, .-test_pc16 

	.globl test_pc16dbl
test_pc16dbl:
  	.reloc test_pc16dbl, R_390_PC16DBL, .-(OFFSET + OFFSET)
	.space 2
  	.size test_pc16dbl, .-test_pc16dbl

