// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: pvalidate
// CHECK: encoding: [0xf2,0x0f,0x01,0xff]
pvalidate

// CHECK: pvalidate
// CHECK: encoding: [0xf2,0x0f,0x01,0xff]
pvalidate	%eax
