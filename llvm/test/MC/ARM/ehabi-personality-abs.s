@ RUN: llvm-mc -triple armv7-linux -filetype obj -o - %s | llvm-objdump --triple=armv7 -D -r - | FileCheck %s

	.text

__aeabi_unwind_cpp_pr0 = 0xdeadbeef

f:
	.fnstart
	bx lr
	.fnend

@@ Regression test: MC does not crash due to the absolute __aeabi_unwind_cpp_pr0.
@@ GNU as and MC currently emit a R_ARM_NONE for this invalid usage.
@ CHECK: R_ARM_NONE __aeabi_unwind_cpp_pr0
