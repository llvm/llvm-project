// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=64 < %t.err %s

// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t.err
// RUN: FileCheck --check-prefix=32 < %t.err %s
// rdar://8204588

// 64: error: ambiguous instructions require an explicit suffix (could be 'cmpb', 'cmpw', 'cmpl', or 'cmpq')
cmp $0, 0(%eax)

// 32: error: register %rax is only available in 64-bit mode
addl $0, 0(%rax)

// 32: error: register %xmm16 is only available in 64-bit mode
// 64: error: register %xmm16 is only available with AVX512
vaddps %xmm16, %xmm0, %xmm0

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
