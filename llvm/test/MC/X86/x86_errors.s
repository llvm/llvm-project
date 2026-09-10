// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=X64 < %t.err %s

// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=X86 < %t.err %s
// rdar://8204588

// X64: error: ambiguous instructions require an explicit suffix (could be 'cmpb', 'cmpw', 'cmpl', or 'cmpq')
cmp $0, 0(%eax)

// X86: error: register %rax is only available in 64-bit mode
addl $0, 0(%rax)

// X86: test.s:8:2: error: invalid instruction mnemonic 'movi'

# 8 "test.s"
 movi $8,%eax

movl 0(%rax), 0(%edx)  // error: invalid operand for instruction

// X86: error: instruction requires: 64-bit mode
sysexitq

// rdar://10710167
// X64: error: expected scale expression
lea (%rsp, %rbp, $4), %rax

// rdar://10423777
// X64: error: base register is 64-bit, but index register is not
movq (%rsi,%ecx),%xmm0

// X64: error: invalid 16-bit base register
movl %eax,(%bp,%si)

// X86: error: scale factor in 16-bit address must be 1
movl %eax,(%bp,%si,2)

// X86: error: invalid 16-bit base register
movl %eax,(%cx)

// X86: error: invalid 16-bit base/index register combination
movl %eax,(%bp,%bx)

// X86: error: 16-bit memory operand may not include only index register
movl %eax,(,%bx)

// X86: error: invalid operand for instruction
outb al, 4

// X86: error: invalid segment register
// X64: error: invalid segment register
movl %eax:0x00, %ebx

// X86: error: invalid operand for instruction
// X64: error: invalid operand for instruction
cmpps $-129, %xmm0, %xmm0

// X86: error: invalid operand for instruction
// X64: error: invalid operand for instruction
cmppd $256, %xmm0, %xmm0

// X86: error: instruction requires: 64-bit mode
jrcxz 1

// X64: error: instruction requires: Not 64-bit mode
jcxz 1

// X86: error: register %cr8 is only available in 64-bit mode
movl %edx, %cr8

// X86: error: register %dr8 is only available in 64-bit mode
movl %edx, %dr8

// X86: error: register %rip is only available in 64-bit mode
// X64: error: %rip can only be used as a base register
mov %rip, %rax

// X86: error: register %rax is only available in 64-bit mode
// X64: error: %rip is not allowed as an index register
mov (%rax,%rip), %rbx

// X86: error: instruction requires: 64-bit mode
ljmpq *(%eax)

// X86: error: register %rax is only available in 64-bit mode
// X64: error: invalid base+index expression
leaq (%rax,%rsp), %rax

// X86: error: invalid base+index expression
// X64: error: invalid base+index expression
leaq (%eax,%esp), %eax

// X86: error: invalid 16-bit base/index register combination
// X64: error: invalid 16-bit base register
lea (%si,%bp), %ax
// X86: error: invalid 16-bit base/index register combination
// X64: error: invalid 16-bit base register
lea (%di,%bp), %ax
// X86: error: invalid 16-bit base/index register combination
// X64: error: invalid 16-bit base register
lea (%si,%bx), %ax
// X86: error: invalid 16-bit base/index register combination
// X64: error: invalid 16-bit base register
lea (%di,%bx), %ax

// X86: error: invalid base+index expression
// X64: error: invalid base+index expression
mov (,%eip), %rbx

// X86: error: invalid base+index expression
// X64: error: invalid base+index expression
mov (%eip,%eax), %rbx

// X86: error: register %rax is only available in 64-bit mode
// X64: error: base register is 64-bit, but index register is not
mov (%rax,%eiz), %ebx

// X86: error: register %riz is only available in 64-bit mode
// X64: error: base register is 32-bit, but index register is not
mov (%eax,%riz), %ebx


// Parse errors from assembler parsing. 

v_ecx = %ecx
v_eax = %eax
v_gs  = %gs
v_imm = 4
$test = %ebx

// X86: 7: error: expected register here
// X64: 7: error: expected register here
mov 4(4), %eax	

// X86: 7: error: expected register here
// X64: 7: error: expected register here
mov 5(v_imm), %eax		
	
// X86: 7: error: invalid register name
// X64: 7: error: invalid register name
mov 6(%v_imm), %eax		
	
// X86: 8: warning: scale factor without index register is ignored
// X64: 8: warning: scale factor without index register is ignored
mov 7(,v_imm), %eax		

// X64: 6: error: expected immediate expression
mov $%eax, %ecx

// X86: 6: error: expected immediate expression
// X64: 6: error: expected immediate expression
mov $v_eax, %ecx

// X86: error: unexpected token in argument list
// X64: error: unexpected token in argument list
mov v_ecx(%eax), %ecx	

// X86: 7: error: invalid operand for instruction
// X64: 7: error: invalid operand for instruction
addb (%dx), %al

// X86: error: instruction requires: 64-bit mode
cqto

// X86: error: instruction requires: 64-bit mode
cltq

// X86: error: instruction requires: 64-bit mode
cmpxchg16b (%eax)

// X86: error: unsupported instruction
// X64: error: unsupported instruction
{rex} vmovdqu32 %xmm0, %xmm0

// X86: error: unsupported instruction
// X64: error: unsupported instruction
{rex2} vmovdqu32 %xmm0, %xmm0

// X86: error: unsupported instruction
// X64: error: unsupported instruction
{vex} vmovdqu32 %xmm0, %xmm0

// X86: error: unsupported instruction
// X64: error: unsupported instruction
{vex2} vmovdqu32 %xmm0, %xmm0

// X86: error: unsupported instruction
// X64: error: unsupported instruction
{vex3} vmovdqu32 %xmm0, %xmm0

// X86: error: unsupported instruction
// X64: error: unsupported instruction
{evex} vmovdqu %xmm0, %xmm0

// X86: 12: error: immediate must be an integer in range [0, 15]
// X64: 12: error: immediate must be an integer in range [0, 15]
vpermil2pd $16, %xmm3, %xmm5, %xmm1, %xmm2

// X86: error: instruction requires: 64-bit mode
pbndkb

// X86: error: register %r16d is only available in 64-bit mode
movl %eax, %r16d
