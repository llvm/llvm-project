// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=64 %s
// RUN: FileCheck --check-prefix=ERR64 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=32 %s
// RUN: FileCheck --check-prefix=ERR32 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown-code16 --show-encoding %s 2> %t.err | FileCheck --check-prefix=16 %s
// RUN: FileCheck --check-prefix=ERR16 < %t.err %s

lodsb
// 64: lodsb (%rsi), %al # encoding: [0xac]
// 32: lodsb (%esi), %al # encoding: [0xac]
// 16: lodsb (%si), %al # encoding: [0xac]

lodsb (%rsi), %al
// 64: lodsb (%rsi), %al # encoding: [0xac]
// ERR32: 64-bit
// ERR16: 64-bit

lodsb (%esi), %al
// 64: lodsb (%esi), %al # encoding: [0x67,0xac]
// 32: lodsb (%esi), %al # encoding: [0xac]
// 16: lodsb (%esi), %al # encoding: [0x67,0xac]

lodsb (%si), %al
// ERR64: invalid 16-bit base register
// 32: lodsb (%si), %al # encoding: [0x67,0xac]
// 16: lodsb (%si), %al # encoding: [0xac]

lodsl %gs:(%esi)
// 64: lodsl %gs:(%esi), %eax # encoding: [0x65,0x67,0xad]
// 32: lodsl %gs:(%esi), %eax # encoding: [0x65,0xad]
// 16: lodsl %gs:(%esi), %eax # encoding: [0x66,0x65,0x67,0xad]

lodsl (%edi), %eax
// ERR64: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lodsl 44(%edi), %eax
// ERR64: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lods (%esi), %ax
// 64: lodsw (%esi), %ax # encoding: [0x66,0x67,0xad]
// 32: lodsw (%esi), %ax # encoding: [0x66,0xad]
// 16: lodsw (%esi), %ax # encoding: [0x67,0xad]

stosw
// 64: stosw %ax, %es:(%rdi) # encoding: [0x66,0xab]
// 32: stosw %ax, %es:(%edi) # encoding: [0x66,0xab]
// 16: stosw %ax, %es:(%di) # encoding: [0xab]

stos %eax, (%edi)
// 64: stosl %eax, %es:(%edi) # encoding: [0x67,0xab]
// 32: stosl %eax, %es:(%edi) # encoding: [0xab]
// 16: stosl %eax, %es:(%edi) # encoding: [0x66,0x67,0xab]

stosb %al, %fs:(%edi)
// ERR64: invalid operand for instruction
// ERR32: invalid operand for instruction
// ERR16: invalid operand for instruction

stosb %al, %es:(%edi)
// 64: stosb %al, %es:(%edi) # encoding: [0x67,0xaa]
// 32: stosb %al, %es:(%edi) # encoding: [0xaa]
// 16: stosb %al, %es:(%edi) # encoding: [0x67,0xaa]

stosq
// 64: stosq %rax, %es:(%rdi) # encoding: [0x48,0xab]
// ERR32: 64-bit
// ERR16: 64-bit

stos %rax, (%edi)
// 64: 	stosq %rax, %es:(%edi) # encoding: [0x48,0x67,0xab]
// ERR32: only available in 64-bit mode
// ERR16: only available in 64-bit mode

scas %es:(%edi), %al
// 64: scasb %es:(%edi), %al # encoding: [0x67,0xae]
// 32: scasb %es:(%edi), %al # encoding: [0xae]
// 16: scasb %es:(%edi), %al # encoding: [0x67,0xae]

scasq %es:(%edi)
// 64: scasq %es:(%edi), %rax # encoding: [0x48,0x67,0xaf]
// ERR32: 64-bit
// ERR16: 64-bit

scasl %es:(%edi), %al
// ERR64: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

scas %es:(%di), %ax
// ERR64: invalid 16-bit base register
// 16: scasw %es:(%di), %ax # encoding: [0xaf]
// 32: scasw %es:(%di), %ax # encoding: [0x66,0x67,0xaf]

cmpsb
// 64: cmpsb %es:(%rdi), (%rsi) # encoding: [0xa6]
// 32: cmpsb %es:(%edi), (%esi) # encoding: [0xa6]
// 16: cmpsb %es:(%di), (%si) # encoding: [0xa6]

cmpsw (%edi), (%esi)
// 64: cmpsw %es:(%edi), (%esi) # encoding: [0x66,0x67,0xa7]
// 32: cmpsw %es:(%edi), (%esi) # encoding: [0x66,0xa7]
// 16: cmpsw %es:(%edi), (%esi) # encoding: [0x67,0xa7]

cmpsb (%di), (%esi)
// ERR64: invalid 16-bit base register
// ERR32: mismatching source and destination
// ERR16: mismatching source and destination

cmpsl %es:(%edi), %ss:(%esi)
// 64: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x36,0x67,0xa7]
// 32: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x36,0xa7]
// 16: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x66,0x36,0x67,0xa7]

cmpsq (%rdi), (%rsi)
// 64: cmpsq %es:(%rdi), (%rsi) # encoding: [0x48,0xa7]
// ERR32: 64-bit
// ERR16: 64-bit

movsb (%esi), (%edi)
// 64: movsb (%esi), %es:(%edi) # encoding: [0x67,0xa4]
// 32: movsb (%esi), %es:(%edi) # encoding: [0xa4]
// 16: movsb (%esi), %es:(%edi) # encoding: [0x67,0xa4]

movsl %gs:(%esi), (%edi)
// 64: movsl %gs:(%esi), %es:(%edi) # encoding: [0x65,0x67,0xa5]
// 32: movsl %gs:(%esi), %es:(%edi) # encoding: [0x65,0xa5]
// 16: movsl %gs:(%esi), %es:(%edi) # encoding: [0x66,0x65,0x67,0xa5]

outsb
// 64: outsb (%rsi), %dx # encoding: [0x6e]
// 32: outsb (%esi), %dx # encoding: [0x6e]
// 16: outsb (%si), %dx # encoding: [0x6e]

outsw %fs:(%esi), %dx
// 64: outsw %fs:(%esi), %dx # encoding: [0x66,0x64,0x67,0x6f]
// 32: outsw %fs:(%esi), %dx # encoding: [0x66,0x64,0x6f]
// 16: outsw %fs:(%esi), %dx # encoding: [0x64,0x67,0x6f]

insw %dx, (%edi)
// 64: insw %dx, %es:(%edi) # encoding: [0x66,0x67,0x6d]
// 32: insw %dx, %es:(%edi) # encoding: [0x66,0x6d]
// 16: insw %dx, %es:(%edi) # encoding: [0x67,0x6d]

insw %dx, (%bx)
// ERR64: invalid 16-bit base register
// 32: insw %dx, %es:(%di) # encoding: [0x66,0x67,0x6d]
// 16: insw %dx, %es:(%di) # encoding: [0x6d]

insw %dx, (%ebx)
// 64: insw %dx, %es:(%edi) # encoding: [0x66,0x67,0x6d]
// 32: insw %dx, %es:(%edi) # encoding: [0x66,0x6d]
// 16: insw %dx, %es:(%edi) # encoding: [0x67,0x6d]

insw %dx, (%rbx)
// 64: insw %dx, %es:(%rdi) # encoding: [0x66,0x6d]
// ERR32: 64-bit
// ERR16: 64-bit

