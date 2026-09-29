// RUN: llvm-mc -triple aarch64 -mattr=+el3 -show-encoding < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -mattr=+v8a -show-encoding < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8r %s  2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// The immediate defaults to zero
// CHECK: dcps3 // encoding: [0x03,0x00,0xa0,0xd4]
dcps3

// CHECK: encoding: [0x83,0x00,0xa0,0xd4]
// CHECK: encoding: [0xe3,0x00,0x00,0xd4]

dcps3 #4
smc #7

// CHECK-ERROR: {{[0-9]+}}:{{[0-9]+}}: error: instruction requires: el3
// CHECK-ERROR-NEXT: dcps3
// CHECK-ERROR-NEXT: ^
// CHECK-ERROR-NEXT: {{[0-9]+}}:{{[0-9]+}}: error: instruction requires: el3
// CHECK-ERROR-NEXT: dcps3 #4
// CHECK-ERROR-NEXT: ^
// CHECK-ERROR-NEXT: {{[0-9]+}}:{{[0-9]+}}: error: instruction requires: el3
// CHECK-ERROR-NEXT: smc #7
// CHECK-ERROR-NEXT: ^
