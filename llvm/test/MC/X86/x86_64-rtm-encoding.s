// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xbegin .L0
// CHECK: encoding: [0xc7,0xf8,A,A,A,A]
	xbegin .L0

// CHECK: xend
// CHECK: encoding: [0x0f,0x01,0xd5]
	xend

// CHECK: xabort
// CHECK: encoding: [0xc6,0xf8,0x0d]
	xabort $13
