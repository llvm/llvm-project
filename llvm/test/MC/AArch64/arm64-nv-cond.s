// RUN: llvm-mc < %s -triple arm64 -mattr=neon -show-encoding | FileCheck %s

fcsel d28,d31,d31,nv
csel x0,x0,x0,nv
ccmp x0,x0,#0,nv
b.nv #0

// CHECK: fcsel   d28, d31, d31, nv       // encoding: [0xfc,0xff,0x7f,0x1e]
// CHECK: csel    x0, x0, x0, nv          // encoding: [0x00,0xf0,0x80,0x9a]
// CHECK: ccmp    x0, x0, #0, nv          // encoding: [0x00,0xf0,0x40,0xfa]
// CHECK: b.nv #0                         // encoding: [0x0f,0x00,0x00,0x54]
