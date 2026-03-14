// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding < %s | FileCheck %s

fadd v0.2d, v5.2d, v6.2d
fadd V0.2d, V5.2d, V6.2d
fadd v0.2d, V5.2d, v6.2d
// CHECK: fadd v0.2d, v5.2d, v6.2d          // encoding: [0xa0,0xd4,0x66,0x4e]
// CHECK: fadd v0.2d, v5.2d, v6.2d          // encoding: [0xa0,0xd4,0x66,0x4e]
// CHECK: fadd v0.2d, v5.2d, v6.2d          // encoding: [0xa0,0xd4,0x66,0x4e]
