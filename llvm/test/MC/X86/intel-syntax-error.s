// RUN: not llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=att %s -o /dev/null 2>&1 | FileCheck %s

// This tests weird forms of Intel and AT&T syntax that gas accepts that we
// don't.  The [no]prefix operand of the syntax directive indicates whether
// registers need a '%' prefix.

.intel_syntax prefix
// CHECK: error: '.intel_syntax prefix' is not supported: registers must not have a '%' prefix in .intel_syntax
_test2:
	mov	DWORD PTR [%esp - 4], 257
.att_syntax noprefix
// CHECK: error: '.att_syntax noprefix' is not supported: registers must have a '%' prefix in .att_syntax
	movl	$257, -4(esp)
