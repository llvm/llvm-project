// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=64 < %t.err %s

// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=32 < %t.err %s
// rdar://8204588

// 64: error: ambiguous instructions require an explicit suffix (could be 'cmpb', 'cmpw', 'cmpl', or 'cmpq')
cmp $0, 0(%eax)

// 32: error: register %rax is only available in 64-bit mode
addl $0, 0(%rax)

// 32: test.s:8:2: error: invalid instruction mnemonic 'movi'

# 8 "test.s"
 movi $8,%eax

movl 0(%rax), 0(%edx)  // error: invalid operand for instruction

// 32: error: instruction requires: 64-bit mode
sysexitq

// rdar://10710167
// 64: error: expected scale expression
lea (%rsp, %rbp, $4), %rax

// rdar://10423777
// 64: error: base register is 64-bit, but index register is not
movq (%rsi,%ecx),%xmm0

// 64: error: invalid 16-bit base register
movl %eax,(%bp,%si)

// 32: error: scale factor in 16-bit address must be 1
movl %eax,(%bp,%si,2)

// 32: error: invalid 16-bit base register
movl %eax,(%cx)

// 32: error: invalid 16-bit base/index register combination
movl %eax,(%bp,%bx)

// 32: error: 16-bit memory operand may not include only index register
movl %eax,(,%bx)

// 32: error: invalid operand for instruction
outb al, 4

// 32: error: invalid segment register
// 64: error: invalid segment register
movl %eax:0x00, %ebx

// 32: error: invalid operand for instruction
// 64: error: invalid operand for instruction
cmpps $-129, %xmm0, %xmm0

// 32: error: invalid operand for instruction
// 64: error: invalid operand for instruction
cmppd $256, %xmm0, %xmm0

// 32: error: instruction requires: 64-bit mode
jrcxz 1

// 64: error: instruction requires: Not 64-bit mode
jcxz 1

// 32: error: register %cr8 is only available in 64-bit mode
movl %edx, %cr8

// 32: error: register %dr8 is only available in 64-bit mode
movl %edx, %dr8

// 32: error: register %rip is only available in 64-bit mode
// 64: error: %rip can only be used as a base register
mov %rip, %rax

// 32: error: register %rax is only available in 64-bit mode
// 64: error: %rip is not allowed as an index register
mov (%rax,%rip), %rbx

// 32: error: instruction requires: 64-bit mode
ljmpq *(%eax)

// 32: error: register %rax is only available in 64-bit mode
// 64: error: invalid base+index expression
leaq (%rax,%rsp), %rax

// 32: error: invalid base+index expression
// 64: error: invalid base+index expression
leaq (%eax,%esp), %eax

// 32: error: invalid 16-bit base/index register combination
// 64: error: invalid 16-bit base register
lea (%si,%bp), %ax
// 32: error: invalid 16-bit base/index register combination
// 64: error: invalid 16-bit base register
lea (%di,%bp), %ax
// 32: error: invalid 16-bit base/index register combination
// 64: error: invalid 16-bit base register
lea (%si,%bx), %ax
// 32: error: invalid 16-bit base/index register combination
// 64: error: invalid 16-bit base register
lea (%di,%bx), %ax

// 32: error: invalid base+index expression
// 64: error: invalid base+index expression
mov (,%eip), %rbx

// 32: error: invalid base+index expression
// 64: error: invalid base+index expression
mov (%eip,%eax), %rbx

// 32: error: register %rax is only available in 64-bit mode
// 64: error: base register is 64-bit, but index register is not
mov (%rax,%eiz), %ebx

// 32: error: register %riz is only available in 64-bit mode
// 64: error: base register is 32-bit, but index register is not
mov (%eax,%riz), %ebx


// Parse errors from assembler parsing. 

v_ecx = %ecx
v_eax = %eax
v_gs  = %gs
v_imm = 4
$test = %ebx

// 32: 7: error: expected register here
// 64: 7: error: expected register here
mov 4(4), %eax	

// 32: 7: error: expected register here
// 64: 7: error: expected register here
mov 5(v_imm), %eax		
	
// 32: 7: error: invalid register name
// 64: 7: error: invalid register name
mov 6(%v_imm), %eax		
	
// 32: 8: warning: scale factor without index register is ignored
// 64: 8: warning: scale factor without index register is ignored
mov 7(,v_imm), %eax		

// 64: 6: error: expected immediate expression
mov $%eax, %ecx

// 32: 6: error: expected immediate expression
// 64: 6: error: expected immediate expression
mov $v_eax, %ecx

// 32: error: unexpected token in argument list
// 64: error: unexpected token in argument list
mov v_ecx(%eax), %ecx	

// 32: 7: error: invalid operand for instruction
// 64: 7: error: invalid operand for instruction
addb (%dx), %al

// 32: error: instruction requires: 64-bit mode
cqto

// 32: error: instruction requires: 64-bit mode
cltq

// 32: error: instruction requires: 64-bit mode
cmpxchg16b (%eax)

// 32: error: unsupported instruction
// 64: error: unsupported instruction
{vex} vmovdqu32 %xmm0, %xmm0

// 32: error: unsupported instruction
// 64: error: unsupported instruction
{vex2} vmovdqu32 %xmm0, %xmm0

// 32: error: unsupported instruction
// 64: error: unsupported instruction
{vex3} vmovdqu32 %xmm0, %xmm0

// 32: error: unsupported instruction
// 64: error: unsupported instruction
{evex} vmovdqu %xmm0, %xmm0

// 32: 12: error: immediate must be an integer in range [0, 15]
// 64: 12: error: immediate must be an integer in range [0, 15]
vpermil2pd $16, %xmm3, %xmm5, %xmm1, %xmm2
