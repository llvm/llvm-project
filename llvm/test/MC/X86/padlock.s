// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

	xstore
// CHECK: xstore
// CHECK: encoding: [0xf3,0x0f,0xa7,0xc0]
