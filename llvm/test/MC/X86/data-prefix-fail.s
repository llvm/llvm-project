// RUN: not llvm-mc -triple x86_64-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=64 %s
// RUN: FileCheck --check-prefix=ERR64 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown --show-encoding %s 2> %t.err | FileCheck --check-prefix=32 %s
// RUN: FileCheck --check-prefix=ERR32 < %t.err %s
// RUN: not llvm-mc -triple i386-unknown-unknown-code16 --show-encoding %s 2> %t.err | FileCheck --check-prefix=16 %s
// RUN: FileCheck --check-prefix=ERR16 < %t.err %s

// ERR64: error: 'data32' is not supported in 64-bit mode
// ERR32: error: redundant data32 prefix
// 16: data32
// 16: encoding: [0x66]
// 16: lgdtw 0
// 16: encoding: [0x0f,0x01,0x16,0x00,0x00]
data32 lgdt 0

// 64: data16
// 64: encoding: [0x66]
// 64: lgdtq 0
// 64: encoding: [0x0f,0x01,0x14,0x25,0x00,0x00,0x00,0x00]
// 32: data16
// 32: encoding: [0x66]
// 32: lgdtl 0
// 32: encoding: [0x0f,0x01,0x15,0x00,0x00,0x00,0x00]
// ERR16: error: redundant data16 prefix
data16 lgdt 0
