// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=X86-64 %s
// RUN: FileCheck --input-file=%t.err %s --check-prefix=ERR64 --implicit-check-not=error:
// RUN: not llvm-mc -triple i386-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=X86-32 %s
// RUN: FileCheck --check-prefix=ERR32 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown-code16 --show-encoding %s 2> %t.err | FileCheck --check-prefix=X86-16 %s
// RUN: FileCheck --check-prefix=ERR16 < %t.err %s

lodsb
// X86-64: lodsb (%rsi), %al # encoding: [0xac]
// X86-32: lodsb (%esi), %al # encoding: [0xac]
// X86-16: lodsb (%si), %al # encoding: [0xac]

lodsb (%rsi), %al
// X86-64: lodsb (%rsi), %al # encoding: [0xac]
// ERR32: 64-bit
// ERR16: 64-bit

lodsb (%esi), %al
// X86-64: lodsb (%esi), %al # encoding: [0x67,0xac]
// X86-32: lodsb (%esi), %al # encoding: [0xac]
// X86-16: lodsb (%esi), %al # encoding: [0x67,0xac]

lodsb (%si), %al
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// X86-32: lodsb (%si), %al # encoding: [0x67,0xac]
// X86-16: lodsb (%si), %al # encoding: [0xac]

lodsl %gs:(%esi)
// X86-64: lodsl %gs:(%esi), %eax # encoding: [0x67,0x65,0xad]
// X86-32: lodsl %gs:(%esi), %eax # encoding: [0x65,0xad]
// X86-16: lodsl %gs:(%esi), %eax # encoding: [0x67,0x65,0x66,0xad]

lodsl (%edi), %eax
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lodsl 44(%edi), %eax
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

lods (%esi), %ax
// X86-64: lodsw (%esi), %ax # encoding: [0x67,0x66,0xad]
// X86-32: lodsw (%esi), %ax # encoding: [0x66,0xad]
// X86-16: lodsw (%esi), %ax # encoding: [0x67,0xad]

stosw
// X86-64: stosw %ax, %es:(%rdi) # encoding: [0x66,0xab]
// X86-32: stosw %ax, %es:(%edi) # encoding: [0x66,0xab]
// X86-16: stosw %ax, %es:(%di) # encoding: [0xab]

stos %eax, (%edi)
// X86-64: stosl %eax, %es:(%edi) # encoding: [0x67,0xab]
// X86-32: stosl %eax, %es:(%edi) # encoding: [0xab]
// X86-16: stosl %eax, %es:(%edi) # encoding: [0x67,0x66,0xab]

stosb %al, %fs:(%edi)
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand for instruction
// ERR32: invalid operand for instruction
// ERR16: invalid operand for instruction

stosb %al, %es:(%edi)
// X86-64: stosb %al, %es:(%edi) # encoding: [0x67,0xaa]
// X86-32: stosb %al, %es:(%edi) # encoding: [0xaa]
// X86-16: stosb %al, %es:(%edi) # encoding: [0x67,0xaa]

stosq
// X86-64: stosq %rax, %es:(%rdi) # encoding: [0x48,0xab]
// ERR32: 64-bit
// ERR16: 64-bit

stos %rax, (%edi)
// X86-64: 	stosq %rax, %es:(%edi) # encoding: [0x67,0x48,0xab]
// ERR32: only available in 64-bit mode
// ERR16: only available in 64-bit mode

scas %es:(%edi), %al
// X86-64: scasb %es:(%edi), %al # encoding: [0x67,0xae]
// X86-32: scasb %es:(%edi), %al # encoding: [0xae]
// X86-16: scasb %es:(%edi), %al # encoding: [0x67,0xae]

scasq %es:(%edi)
// X86-64: scasq %es:(%edi), %rax # encoding: [0x67,0x48,0xaf]
// ERR32: 64-bit
// ERR16: 64-bit

scasl %es:(%edi), %al
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid operand
// ERR32: invalid operand
// ERR16: invalid operand

scas %es:(%di), %ax
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// X86-16: scasw %es:(%di), %ax # encoding: [0xaf]
// X86-32: scasw %es:(%di), %ax # encoding: [0x67,0x66,0xaf]

cmpsb
// X86-64: cmpsb %es:(%rdi), (%rsi) # encoding: [0xa6]
// X86-32: cmpsb %es:(%edi), (%esi) # encoding: [0xa6]
// X86-16: cmpsb %es:(%di), (%si) # encoding: [0xa6]

cmpsw (%edi), (%esi)
// X86-64: cmpsw %es:(%edi), (%esi) # encoding: [0x67,0x66,0xa7]
// X86-32: cmpsw %es:(%edi), (%esi) # encoding: [0x66,0xa7]
// X86-16: cmpsw %es:(%edi), (%esi) # encoding: [0x67,0xa7]

cmpsb (%di), (%esi)
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// ERR32: mismatching source and destination
// ERR16: mismatching source and destination

cmpsl %es:(%edi), %ss:(%esi)
// X86-64: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x67,0x36,0xa7]
// X86-32: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x36,0xa7]
// X86-16: cmpsl %es:(%edi), %ss:(%esi) # encoding: [0x67,0x36,0x66,0xa7]

cmpsq (%rdi), (%rsi)
// X86-64: cmpsq %es:(%rdi), (%rsi) # encoding: [0x48,0xa7]
// ERR32: 64-bit
// ERR16: 64-bit

movsb (%esi), (%edi)
// X86-64: movsb (%esi), %es:(%edi) # encoding: [0x67,0xa4]
// X86-32: movsb (%esi), %es:(%edi) # encoding: [0xa4]
// X86-16: movsb (%esi), %es:(%edi) # encoding: [0x67,0xa4]

movsl %gs:(%esi), (%edi)
// X86-64: movsl %gs:(%esi), %es:(%edi) # encoding: [0x67,0x65,0xa5]
// X86-32: movsl %gs:(%esi), %es:(%edi) # encoding: [0x65,0xa5]
// X86-16: movsl %gs:(%esi), %es:(%edi) # encoding: [0x67,0x65,0x66,0xa5]

outsb
// X86-64: outsb (%rsi), %dx # encoding: [0x6e]
// X86-32: outsb (%esi), %dx # encoding: [0x6e]
// X86-16: outsb (%si), %dx # encoding: [0x6e]

outsw %fs:(%esi), %dx
// X86-64: outsw %fs:(%esi), %dx # encoding: [0x67,0x64,0x66,0x6f]
// X86-32: outsw %fs:(%esi), %dx # encoding: [0x64,0x66,0x6f]
// X86-16: outsw %fs:(%esi), %dx # encoding: [0x67,0x64,0x6f]

insw %dx, (%edi)
// X86-64: insw %dx, %es:(%edi) # encoding: [0x67,0x66,0x6d]
// X86-32: insw %dx, %es:(%edi) # encoding: [0x66,0x6d]
// X86-16: insw %dx, %es:(%edi) # encoding: [0x67,0x6d]

insw %dx, (%bx)
// ERR64: [[#@LINE-1]]:[[#]]: error: invalid 16-bit base register
// X86-32: insw %dx, %es:(%di) # encoding: [0x67,0x66,0x6d]
// X86-16: insw %dx, %es:(%di) # encoding: [0x6d]

insw %dx, (%ebx)
// X86-64: insw %dx, %es:(%edi) # encoding: [0x67,0x66,0x6d]
// X86-32: insw %dx, %es:(%edi) # encoding: [0x66,0x6d]
// X86-16: insw %dx, %es:(%edi) # encoding: [0x67,0x6d]

insw %dx, (%rbx)
// X86-64: insw %dx, %es:(%rdi) # encoding: [0x66,0x6d]
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
// X86-64: movdir64b (%eip), %ebx # encoding: [0x67,0x66,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

movdir64b (%rip), %rbx
// X86-64: movdir64b (%rip), %rbx # encoding: [0x66,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

movdir64b 291(%esi, %eiz, 4), %ebx
// X86-64: movdir64b 291(%esi,%eiz,4), %ebx # encoding: [0x67,0x66,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
// X86-32: movdir64b 291(%esi,%eiz,4), %ebx # encoding: [0x66,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

movdir64b 291(%rsi, %riz, 4), %rbx
// X86-64: movdir64b 291(%rsi,%riz,4), %rbx # encoding: [0x66,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

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
// X86-64: enqcmd (%eip), %ebx # encoding: [0x67,0xf2,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmd (%rip), %rbx
// X86-64: enqcmd (%rip), %rbx # encoding: [0xf2,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmd 291(%esi, %eiz, 4), %ebx
// X86-64: enqcmd 291(%esi,%eiz,4), %ebx # encoding: [0x67,0xf2,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
// X86-32: enqcmd 291(%esi,%eiz,4), %ebx # encoding: [0xf2,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

enqcmd 291(%rsi, %riz, 4), %rbx
// X86-64: enqcmd 291(%rsi,%riz,4), %rbx # encoding: [0xf2,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

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
// X86-64: enqcmds (%eip), %ebx # encoding: [0x67,0xf3,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmds (%rip), %rbx
// X86-64: enqcmds (%rip), %rbx # encoding: [0xf3,0x0f,0x38,0xf8,0x1d,0x00,0x00,0x00,0x00]

enqcmds 291(%esi, %eiz, 4), %ebx
// X86-64: enqcmds 291(%esi,%eiz,4), %ebx # encoding: [0x67,0xf3,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
// X86-32: enqcmds 291(%esi,%eiz,4), %ebx # encoding: [0xf3,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]

enqcmds 291(%rsi, %riz, 4), %rbx
// X86-64: enqcmds 291(%rsi,%riz,4), %rbx # encoding: [0xf3,0x0f,0x38,0xf8,0x9c,0xa6,0x23,0x01,0x00,0x00]
