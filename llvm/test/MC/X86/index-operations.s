// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=64 %s
// RUN: FileCheck --input-file=%t.err %s --check-prefix=ERR64 --implicit-check-not=error:
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
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// 32: lodsb (%si), %al # encoding: [0x67,0xac]
// 16: lodsb (%si), %al # encoding: [0xac]

lodsl %gs:(%esi)
// 64: lodsl %gs:(%esi), %eax # encoding: [0x67,0x65,0xad]
// 32: lodsl %gs:(%esi), %eax # encoding: [0x65,0xad]
// 16: lodsl %gs:(%esi), %eax # encoding: [0x67,0x65,0x66,0xad]

lodsl (%edi), %eax
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lodsl 44(%edi), %eax
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lods (%esi), %ax
// 64: lodsw (%esi), %ax # encoding: [0x67,0x66,0xad]
// 32: lodsw (%esi), %ax # encoding: [0x66,0xad]
// 16: lodsw (%esi), %ax # encoding: [0x67,0xad]

stosw
// 64: stosw %ax, %es:(%rdi) # encoding: [0x66,0xab]
// 32: stosw %ax, %es:(%edi) # encoding: [0x66,0xab]
// 16: stosw %ax, %es:(%di) # encoding: [0xab]

stos %eax, (%edi)
// 64: stosl %eax, %es:(%edi) # encoding: [0x67,0xab]
// 32: stosl %eax, %es:(%edi) # encoding: [0xab]
// 16: stosl %eax, %es:(%edi) # encoding: [0x67,0x66,0xab]

stosb %al, %fs:(%edi)
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand for instruction
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
// 64: 	stosq %rax, %es:(%edi) # encoding: [0x67,0x48,0xab]
// ERR32: only available in 64-bit mode
// ERR16: only available in 64-bit mode

scas %es:(%edi), %al
// 64: scasb %es:(%edi), %al # encoding: [0x67,0xae]
// 32: scasb %es:(%edi), %al # encoding: [0xae]
// 16: scasb %es:(%edi), %al # encoding: [0x67,0xae]

scasq %es:(%edi)
// 64: scasq %es:(%edi), %rax # encoding: [0x67,0x48,0xaf]
// ERR32: 64-bit
// ERR16: 64-bit

scasl %es:(%edi), %al
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

scas %es:(%di), %ax
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// 16: scasw %es:(%di), %ax # encoding: [0xaf]
// 32: scasw %es:(%di), %ax # encoding: [0x67,0x66,0xaf]

cmpsb
// 64: cmpsb %es:(%rdi), (%rsi) # encoding: [0xa6]
// 32: cmpsb %es:(%edi), (%esi) # encoding: [0xa6]
// 16: cmpsb %es:(%di), (%si) # encoding: [0xa6]

cmpsw (%edi), (%esi)
// 64: cmpsw %es:(%edi), (%esi) # encoding: [0x67,0x66,0xa7]
// 32: cmpsw %es:(%edi), (%esi) # encoding: [0x66,0xa7]
// 16: cmpsw %es:(%edi), (%esi) # encoding: [0x67,0xa7]

cmpsb (%di), (%esi)
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// ERR32: mismatching source and destination
// ERR16: mismatching source and destination

cmpsl %es:(%edi), %ss:(%esi)
// 64: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x67,0x36,0xa7]
// 32: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x36,0xa7]
// 16: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x67,0x36,0x66,0xa7]

cmpsq (%rdi), (%rsi)
// 64: cmpsq %es:(%rdi), (%rsi) # encoding: [0x48,0xa7]
// ERR32: 64-bit
// ERR16: 64-bit

movsb (%esi), (%edi)
// 64: movsb (%esi), %es:(%edi) # encoding: [0x67,0xa4]
// 32: movsb (%esi), %es:(%edi) # encoding: [0xa4]
// 16: movsb (%esi), %es:(%edi) # encoding: [0x67,0xa4]

movsl %gs:(%esi), (%edi)
// 64: movsl %gs:(%esi), %es:(%edi) # encoding: [0x67,0x65,0xa5]
// 32: movsl %gs:(%esi), %es:(%edi) # encoding: [0x65,0xa5]
// 16: movsl %gs:(%esi), %es:(%edi) # encoding: [0x67,0x65,0x66,0xa5]

outsb
// 64: outsb (%rsi), %dx # encoding: [0x6e]
// 32: outsb (%esi), %dx # encoding: [0x6e]
// 16: outsb (%si), %dx # encoding: [0x6e]

outsw %fs:(%esi), %dx
// 64: outsw %fs:(%esi), %dx # encoding: [0x67,0x64,0x66,0x6f]
// 32: outsw %fs:(%esi), %dx # encoding: [0x64,0x66,0x6f]
// 16: outsw %fs:(%esi), %dx # encoding: [0x67,0x64,0x6f]

insw %dx, (%edi)
// 64: insw %dx, %es:(%edi) # encoding: [0x67,0x66,0x6d]
// 32: insw %dx, %es:(%edi) # encoding: [0x66,0x6d]
// 16: insw %dx, %es:(%edi) # encoding: [0x67,0x6d]

insw %dx, (%bx)
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// 32: insw %dx, %es:(%di) # encoding: [0x67,0x66,0x6d]
// 16: insw %dx, %es:(%di) # encoding: [0x6d]

insw %dx, (%ebx)
// 64: insw %dx, %es:(%edi) # encoding: [0x67,0x66,0x6d]
// 32: insw %dx, %es:(%edi) # encoding: [0x66,0x6d]
// 16: insw %dx, %es:(%edi) # encoding: [0x67,0x6d]

insw %dx, (%rbx)
// 64: insw %dx, %es:(%rdi) # encoding: [0x66,0x6d]
// ERR32: 64-bit
// ERR16: 64-bit

movdir64b	291(%si), %ecx
// ERR64: error: invalid 16-bit base register
// ERR32: invalid operand
// ERR16: invalid operand

movdir64b	291(%esi), %cx
// ERR64: error: invalid operand for instruction
// ERR32: invalid operand
// ERR16: invalid operand

movdir64b (%rdx), %r15d
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand

movdir64b (%edx), %r15
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand

movdir64b (%eip), %ebx
// 64: movdir64b (%eip), %ebx # encoding: [0x67,0x66,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

movdir64b (%rip), %rbx
// 64: movdir64b (%rip), %rbx # encoding: [0x66,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

movdir64b 291(%esi, %eiz, 4), %ebx
// 64: movdir64b 291(%esi,%eiz,4), %ebx # encoding: [0x67,0x66,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
// 32: movdir64b 291(%esi,%eiz,4), %ebx # encoding: [0x66,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

movdir64b 291(%rsi, %riz, 4), %rbx
// 64: movdir64b 291(%rsi,%riz,4), %rbx # encoding: [0x66,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

enqcmd	291(%si), %ecx
// ERR64: error: invalid 16-bit base register
// ERR32: invalid operand
// ERR16: invalid operand

enqcmd	291(%esi), %cx
// ERR64: error: invalid operand for instruction
// ERR32: invalid operand
// ERR16: invalid operand

enqcmd (%rdx), %r15d
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand

enqcmd (%edx), %r15
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand

enqcmd (%eip), %ebx
// 64: enqcmd (%eip), %ebx # encoding: [0x67,0xf2,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmd (%rip), %rbx
// 64: enqcmd (%rip), %rbx # encoding: [0xf2,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmd 291(%esi, %eiz, 4), %ebx
// 64: enqcmd 291(%esi,%eiz,4), %ebx # encoding: [0x67,0xf2,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
// 32: enqcmd 291(%esi,%eiz,4), %ebx # encoding: [0xf2,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

enqcmd 291(%rsi, %riz, 4), %rbx
// 64: enqcmd 291(%rsi,%riz,4), %rbx # encoding: [0xf2,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

enqcmds	291(%si), %ecx
// ERR64: error: invalid 16-bit base register
// ERR32: invalid operand
// ERR16: invalid operand

enqcmds	291(%esi), %cx
// ERR64: error: invalid operand for instruction
// ERR32: invalid operand
// ERR16: invalid operand

enqcmds (%rdx), %r15d
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand

enqcmds (%edx), %r15
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand

enqcmds (%eip), %ebx
// 64: enqcmds (%eip), %ebx # encoding: [0x67,0xf3,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmds (%rip), %rbx
// 64: enqcmds (%rip), %rbx # encoding: [0xf3,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmds 291(%esi, %eiz, 4), %ebx
// 64: enqcmds 291(%esi,%eiz,4), %ebx # encoding: [0x67,0xf3,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
// 32: enqcmds 291(%esi,%eiz,4), %ebx # encoding: [0xf3,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

enqcmds 291(%rsi, %riz, 4), %rbx
// 64: enqcmds 291(%rsi,%riz,4), %rbx # encoding: [0xf3,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
