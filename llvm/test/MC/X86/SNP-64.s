// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rmpupdate
// CHECK: encoding: [0xf2,0x0f,0x01,0xfe]
rmpupdate %rax, %rcx

// CHECK: psmash
// CHECK: encoding: [0xf3,0x0f,0x01,0xff]
psmash %rax

// CHECK: pvalidate
// CHECK: encoding: [0xf2,0x0f,0x01,0xff]
pvalidate %rax, %rcx, %rdx

// CHECK: rmpadjust
// CHECK: encoding: [0xf3,0x0f,0x01,0xfe]
rmpadjust %rax, %rcx, %rdx 

// CHECK: rmpquery
// CHECK: encoding: [0xf3,0x0f,0x01,0xfd]
rmpquery %rax, %rdx

// CHECK: rmpupdate
// CHECK: encoding: [0xf2,0x0f,0x01,0xfe]
rmpupdate

// CHECK: psmash
// CHECK: encoding: [0xf3,0x0f,0x01,0xff]
psmash

// CHECK: pvalidate
// CHECK: encoding: [0xf2,0x0f,0x01,0xff]
pvalidate

// CHECK: rmpadjust
// CHECK: encoding: [0xf3,0x0f,0x01,0xfe]
rmpadjust

// CHECK: rmpquery
// CHECK: encoding: [0xf3,0x0f,0x01,0xfd]
rmpquery
