// RUN: not llvm-mc -triple i386-unknown-unknown-code16 -x86-asm-syntax=intel --show-encoding %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=ERR < %t.err %s

// A data32 prefix makes an unsized push of an immediate 32-bit, matching the
// AT&T behaviour of the same instruction.

// CHECK: push 8
// CHECK-SAME: encoding: [0x66,0x6a,0x08]
data32 push 8

// CHECK: push 4660
// CHECK-SAME: encoding: [0x66,0x68,0x34,0x12,0x00,0x00]
data32 push 0x1234

// CHECK: push eax
// CHECK-SAME: encoding: [0x66,0x50]
data32 push eax

// Without the prefix the operand size still comes from the mode.

// CHECK: push 8
// CHECK-SAME: encoding: [0x6a,0x08]
push 8

// The prefix applies to one instruction only, so 16-bit mode has to be back in
// effect for whatever follows.

// CHECK: push 4660
// CHECK-SAME: encoding: [0x68,0x34,0x12]
push 0x1234

// The same holds when the prefixed instruction fails to match, otherwise the
// 32-bit mode used for matching leaks into the rest of the file.

// ERR: error: invalid instruction mnemonic 'nosuchinsn'
data32 nosuchinsn

// CHECK: push 8
// CHECK-SAME: encoding: [0x6a,0x08]
push 8
