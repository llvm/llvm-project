// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=X86-64 %s
// RUN: FileCheck --check-prefix=ERR64 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=X86-32 %s
// RUN: FileCheck --check-prefix=ERR32 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown-code16 --show-encoding %s 2> %t.err | FileCheck --check-prefix=X86-16 %s
// RUN: FileCheck --check-prefix=ERR16 < %t.err %s

	ret
// X86-64: retq
// X86-64: encoding: [0xc3]
// X86-32: retl
// X86-32: encoding: [0xc3]
// X86-16: retw
// X86-16: encoding: [0xc3]
	retw
// X86-64: retw
// X86-64: encoding: [0x66,0xc3]
// X86-32: retw
// X86-32: encoding: [0x66,0xc3]
// X86-16: retw
// X86-16: encoding: [0xc3]
	retl
// ERR64: error: instruction requires: Not 64-bit mode
// X86-32: retl
// X86-32: encoding: [0xc3]
// X86-16: retl
// X86-16: encoding: [0x66,0xc3]
	retq
// X86-64: retq
// X86-64: encoding: [0xc3]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode

	ret $0
// X86-64: retq $0
// X86-64: encoding: [0xc2,0x00,0x00]
// X86-32: retl $0
// X86-32: encoding: [0xc2,0x00,0x00]
// X86-16: retw $0
// X86-16: encoding: [0xc2,0x00,0x00]
	retw $0
// X86-64: retw $0
// X86-64: encoding: [0x66,0xc2,0x00,0x00]
// X86-32: retw $0
// X86-32: encoding: [0x66,0xc2,0x00,0x00]
// X86-16: retw $0
// X86-16: encoding: [0xc2,0x00,0x00]
	retl $0
// ERR64: error: instruction requires: Not 64-bit mode
// X86-32: retl $0
// X86-32: encoding: [0xc2,0x00,0x00]
// X86-16: retl $0
// X86-16: encoding: [0x66,0xc2,0x00,0x00]
	retq $0
// X86-64: retq $0
// X86-64: encoding: [0xc2,0x00,0x00]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode

	retn
// X86-64: retq
// X86-64: encoding: [0xc3]
// X86-32: retl
// X86-32: encoding: [0xc3]
// X86-16: retw
// X86-16: encoding: [0xc3]

  retn $0
// X86-64: retq $0
// X86-64: encoding: [0xc2,0x00,0x00]
// X86-32: retl $0
// X86-32: encoding: [0xc2,0x00,0x00]
// X86-16: retw $0
// X86-16: encoding: [0xc2,0x00,0x00]

	lret
// X86-64: lretl
// X86-64: encoding: [0xcb]
// X86-32: lretl
// X86-32: encoding: [0xcb]
// X86-16: lretw
// X86-16: encoding: [0xcb]
	lretw
// X86-64: lretw
// X86-64: encoding: [0x66,0xcb]
// X86-32: lretw
// X86-32: encoding: [0x66,0xcb]
// X86-16: lretw
// X86-16: encoding: [0xcb]
	lretl
// X86-64: lretl
// X86-64: encoding: [0xcb]
// X86-32: lretl
// X86-32: encoding: [0xcb]
// X86-16: lretl
// X86-16: encoding: [0x66,0xcb]
	lretq
// X86-64: lretq
// X86-64: encoding: [0x48,0xcb]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode

	lret $0
// X86-64: lretl $0
// X86-64: encoding: [0xca,0x00,0x00]
// X86-32: lretl $0
// X86-32: encoding: [0xca,0x00,0x00]
// X86-16: lretw $0
// X86-16: encoding: [0xca,0x00,0x00]
	lretw $0
// X86-64: lretw $0
// X86-64: encoding: [0x66,0xca,0x00,0x00]
// X86-32: lretw $0
// X86-32: encoding: [0x66,0xca,0x00,0x00]
// X86-16: lretw $0
// X86-16: encoding: [0xca,0x00,0x00]
	lretl $0
// X86-64: lretl $0
// X86-64: encoding: [0xca,0x00,0x00]
// X86-32: lretl $0
// X86-32: encoding: [0xca,0x00,0x00]
// X86-16: lretl $0
// X86-16: encoding: [0x66,0xca,0x00,0x00]
	lretq $0
// X86-64: lretq $0
// X86-64: encoding: [0x48,0xca,0x00,0x00]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode


