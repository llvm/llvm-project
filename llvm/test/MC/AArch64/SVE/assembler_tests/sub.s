// RUN: llvm-mc -triple=aarch64-none-linux-gnu -show-encoding -mattr=+sve < %s | FileCheck %s
// RUN: not llvm-mc -triple=aarch64-none-linux-gnu -show-encoding -mattr=-sve 2>&1 < %s | FileCheck --check-prefix=CHECK-ERROR %s
sub     z0.h, z0.h, z0.h  // 00000100-01100000-00000100-00000000
// CHECK: sub     z0.h, z0.h, z0.h // encoding: [0x00,0x04,0x60,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-01100000-00000100-00000000
sub     z21.b, z10.b, z21.b  // 00000100-00110101-00000101-01010101
// CHECK: sub     z21.b, z10.b, z21.b // encoding: [0x55,0x05,0x35,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-00110101-00000101-01010101
sub     z31.h, z31.h, z31.h  // 00000100-01111111-00000111-11111111
// CHECK: sub     z31.h, z31.h, z31.h // encoding: [0xff,0x07,0x7f,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-01111111-00000111-11111111
sub     z21.h, z10.h, z21.h  // 00000100-01110101-00000101-01010101
// CHECK: sub     z21.h, z10.h, z21.h // encoding: [0x55,0x05,0x75,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-01110101-00000101-01010101
sub     z31.b, z31.b, z31.b  // 00000100-00111111-00000111-11111111
// CHECK: sub     z31.b, z31.b, z31.b // encoding: [0xff,0x07,0x3f,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-00111111-00000111-11111111
sub     z0.s, z0.s, z0.s  // 00000100-10100000-00000100-00000000
// CHECK: sub     z0.s, z0.s, z0.s // encoding: [0x00,0x04,0xa0,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-10100000-00000100-00000000
sub     z23.b, z13.b, z8.b  // 00000100-00101000-00000101-10110111
// CHECK: sub     z23.b, z13.b, z8.b // encoding: [0xb7,0x05,0x28,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-00101000-00000101-10110111
sub     z21.d, z10.d, z21.d  // 00000100-11110101-00000101-01010101
// CHECK: sub     z21.d, z10.d, z21.d // encoding: [0x55,0x05,0xf5,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-11110101-00000101-01010101
sub     z21.s, z10.s, z21.s  // 00000100-10110101-00000101-01010101
// CHECK: sub     z21.s, z10.s, z21.s // encoding: [0x55,0x05,0xb5,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-10110101-00000101-01010101
sub     z0.b, z0.b, z0.b  // 00000100-00100000-00000100-00000000
// CHECK: sub     z0.b, z0.b, z0.b // encoding: [0x00,0x04,0x20,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-00100000-00000100-00000000
sub     z23.d, z13.d, z8.d  // 00000100-11101000-00000101-10110111
// CHECK: sub     z23.d, z13.d, z8.d // encoding: [0xb7,0x05,0xe8,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-11101000-00000101-10110111
sub     z23.s, z13.s, z8.s  // 00000100-10101000-00000101-10110111
// CHECK: sub     z23.s, z13.s, z8.s // encoding: [0xb7,0x05,0xa8,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-10101000-00000101-10110111
sub     z31.d, z31.d, z31.d  // 00000100-11111111-00000111-11111111
// CHECK: sub     z31.d, z31.d, z31.d // encoding: [0xff,0x07,0xff,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-11111111-00000111-11111111
sub     z23.h, z13.h, z8.h  // 00000100-01101000-00000101-10110111
// CHECK: sub     z23.h, z13.h, z8.h // encoding: [0xb7,0x05,0x68,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-01101000-00000101-10110111
sub     z0.d, z0.d, z0.d  // 00000100-11100000-00000100-00000000
// CHECK: sub     z0.d, z0.d, z0.d // encoding: [0x00,0x04,0xe0,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-11100000-00000100-00000000
sub     z31.s, z31.s, z31.s  // 00000100-10111111-00000111-11111111
// CHECK: sub     z31.s, z31.s, z31.s // encoding: [0xff,0x07,0xbf,0x04]
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: 00000100-10111111-00000111-11111111
