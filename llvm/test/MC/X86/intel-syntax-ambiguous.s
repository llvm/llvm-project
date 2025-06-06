// RUN: not llvm-mc -triple i686-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

.intel_syntax

// Basic case of ambiguity for inc.

inc [eax]
// CHECK: error: ambiguous operand size for instruction 'inc'
inc dword ptr [eax]
inc word ptr [eax]
inc byte ptr [eax]
// CHECK-NOT: error:

// Other ambiguous instructions.  Anything that doesn't take a register,
// basically.

dec [eax]
// CHECK: error: ambiguous operand size for instruction 'dec'
mov [eax], 1
// CHECK: error: ambiguous operand size for instruction 'mov'
and [eax], 0
// CHECK: error: ambiguous operand size for instruction 'and'
or [eax], 1
// CHECK: error: ambiguous operand size for instruction 'or'
add [eax], 1
// CHECK: error: ambiguous operand size for instruction 'add'
sub [eax], 1
// CHECK: error: ambiguous operand size for instruction 'sub'

// gas assumes these instructions are pointer-sized by default, and we follow
// suit.
push [eax]
pop [eax]
call [eax]
jmp [eax]
// CHECK-NOT: error:

add byte ptr [eax], eax
// CHECK: error: invalid operand for instruction

add byte ptr [eax], eax
// CHECK: error: invalid operand for instruction

add rax, 3
// CHECK: error: register %rax is only available in 64-bit mode

fadd   "?half@?0??bar@@YAXXZ@4NA"
// CHECK: error: ambiguous operand size for instruction 'fadd'

// Instruction line with PTR inside check that they don't accept register as memory.

// CHECK:  error: expected memory operand after 'ptr', found register operand instead
// CHECK: andps xmm1, xmmword ptr xmm1
andps xmm1, xmmword ptr xmm1
// CHECK:  error: expected memory operand after 'ptr', found register operand instead
// CHECK: andps xmmword ptr xmm1, xmm1
andps xmmword ptr xmm1, xmm1
// CHECK:  error: expected memory operand after 'ptr', found register operand instead
// CHECK: mov dword ptr eax, ebx
mov dword ptr eax, ebx
