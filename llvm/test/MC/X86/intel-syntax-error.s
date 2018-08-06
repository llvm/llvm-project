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

//CHECK: error: cannot use more than one symbol in memory operand
mov eax, DWORD PTR arr[i]
//CHECK: error: rip can only be used as a base register
.code64
mov rax, rip
//CHECK: error: invalid base+index expression
mov rbx, [rax+rip]
//CHECK: error: scale factor in address must be 1, 2, 4 or 8
lea RDX, [unknown_number * RAX + RBX + _foo]
//CHECK: error: BaseReg/IndexReg already set!
lea RDX, [4 * RAX + 27 * RBX + _pat]
//CHECK: error: unexpected bracket encountered
lea RDX, [[arr]
//CHECK: error: unexpected bracket encountered
lea RDX, [arr[]

.intel_syntax

// CHECK: error: invalid operand for instruction
punpcklbw mm0, qword ptr [rsp]
// CHECK: error: invalid operand for instruction
punpcklwd mm0, word ptr [rsp]
// CHECK: error: invalid operand for instruction
punpckldq mm0, qword ptr [rsp]

// CHECK: error: invalid 16-bit base register
lea bx, [ax]

// CHECK: invalid base+index expression
lea eax, [eax+esp*1]

// CHECK: 16-bit addresses cannot have a scale
lea ax, [bx+si*1]
