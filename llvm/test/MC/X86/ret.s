// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=64 %s
// RUN: FileCheck --check-prefix=ERR64 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=32 %s
// RUN: FileCheck --check-prefix=ERR32 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown-code16 --show-encoding %s 2> %t.err | FileCheck --check-prefix=16 %s
// RUN: FileCheck --check-prefix=ERR16 < %t.err %s

	ret
// 64: retq
// 64: encoding: [0xc3]
// 32: retl
// 32: encoding: [0xc3]
// 16: retw
// 16: encoding: [0xc3]
	retw
// 64: retw
// 64: encoding: [0x66,0xc3]
// 32: retw
// 32: encoding: [0x66,0xc3]
// 16: retw
// 16: encoding: [0xc3]
	retl
// ERR64: error: instruction requires: Not 64-bit mode
// 32: retl
// 32: encoding: [0xc3]
// 16: retl
// 16: encoding: [0x66,0xc3]
	retq
// 64: retq
// 64: encoding: [0xc3]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode

	ret $0
// 64: retq $0
// 64: encoding: [0xc2,0x00,0x00]
// 32: retl $0
// 32: encoding: [0xc2,0x00,0x00]
// 16: retw $0
// 16: encoding: [0xc2,0x00,0x00]
	retw $0
// 64: retw $0
// 64: encoding: [0x66,0xc2,0x00,0x00]
// 32: retw $0
// 32: encoding: [0x66,0xc2,0x00,0x00]
// 16: retw $0
// 16: encoding: [0xc2,0x00,0x00]
	retl $0
// ERR64: error: instruction requires: Not 64-bit mode
// 32: retl $0
// 32: encoding: [0xc2,0x00,0x00]
// 16: retl $0
// 16: encoding: [0x66,0xc2,0x00,0x00]
	retq $0
// 64: retq $0
// 64: encoding: [0xc2,0x00,0x00]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode

	retn
// 64: retq
// 64: encoding: [0xc3]
// 32: retl
// 32: encoding: [0xc3]
// 16: retw
// 16: encoding: [0xc3]

  retn $0
// 64: retq $0
// 64: encoding: [0xc2,0x00,0x00]
// 32: retl $0
// 32: encoding: [0xc2,0x00,0x00]
// 16: retw $0
// 16: encoding: [0xc2,0x00,0x00]

	lret
// 64: lretl
// 64: encoding: [0xcb]
// 32: lretl
// 32: encoding: [0xcb]
// 16: lretw
// 16: encoding: [0xcb]
	lretw
// 64: lretw
// 64: encoding: [0x66,0xcb]
// 32: lretw
// 32: encoding: [0x66,0xcb]
// 16: lretw
// 16: encoding: [0xcb]
	lretl
// 64: lretl
// 64: encoding: [0xcb]
// 32: lretl
// 32: encoding: [0xcb]
// 16: lretl
// 16: encoding: [0x66,0xcb]
	lretq
// 64: lretq
// 64: encoding: [0x48,0xcb]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode

	lret $0
// 64: lretl $0
// 64: encoding: [0xca,0x00,0x00]
// 32: lretl $0
// 32: encoding: [0xca,0x00,0x00]
// 16: lretw $0
// 16: encoding: [0xca,0x00,0x00]
	lretw $0
// 64: lretw $0
// 64: encoding: [0x66,0xca,0x00,0x00]
// 32: lretw $0
// 32: encoding: [0x66,0xca,0x00,0x00]
// 16: lretw $0
// 16: encoding: [0xca,0x00,0x00]
	lretl $0
// 64: lretl $0
// 64: encoding: [0xca,0x00,0x00]
// 32: lretl $0
// 32: encoding: [0xca,0x00,0x00]
// 16: lretl $0
// 16: encoding: [0x66,0xca,0x00,0x00]
	lretq $0
// 64: lretq $0
// 64: encoding: [0x48,0xca,0x00,0x00]
// ERR32: error: instruction requires: 64-bit mode
// ERR16: error: instruction requires: 64-bit mode


