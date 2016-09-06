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


.intel_syntax noprefix

.global arr
.global i
.set FOO, 2
//CHECK-STDERR: error: cannot use base register with variable reference
mov eax, DWORD PTR arr[ebp + 1 + (2 * 5) - 3 + 1<<1]
//CHECK-STDERR: error: cannot use index register with variable reference
mov eax, DWORD PTR arr[esi*4]
//CHECK-STDERR: error: cannot use more than one symbol in memory operand
mov eax, DWORD PTR arr[i]
