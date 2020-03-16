@ RUN: llvm-mc -triple armv7-linux -filetype obj -o - %s | llvm-objdump --triple=armv7 -D -r - | FileCheck %s

	.text

__aeabi_unwind_cpp_pr0 = 0xdeadbeef

f:
	.fnstart
	bx lr
	.fnend

@ CHECK: R_ARM_NONE __aeabi_unwind_cpp_pr0

