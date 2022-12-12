// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr +lse128 %s 2>%t | FileCheck %s
// RUN: FileCheck %s --input-file=%t --check-prefix=ERROR-INVALID-OP
// RUN: not llvm-mc -triple aarch64 -show-encoding %s 2>&1 | FileCheck --check-prefix=ERROR-NO-LSE128 %s

ldclrp   x1, x2, [x11]
// CHECK: ldclrp x1, x2, [x11]                   // encoding: [0x61,0x11,0x22,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrp   x21, x22, [sp]
// CHECK: ldclrp x21, x22, [sp]                  // encoding: [0xf5,0x13,0x36,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrpa  x1, x2, [x11]
// CHECK: ldclrpa x1, x2, [x11]                   // encoding: [0x61,0x11,0xa2,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrpa  x21, x22, [sp]
// CHECK: ldclrpa x21, x22, [sp]                  // encoding: [0xf5,0x13,0xb6,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrpal x1, x2, [x11]
// CHECK: ldclrpal x1, x2, [x11]                   // encoding: [0x61,0x11,0xe2,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrpal x21, x22, [sp]
// CHECK: ldclrpal x21, x22, [sp]                  // encoding: [0xf5,0x13,0xf6,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrpl  x1, x2, [x11]
// CHECK: ldclrpl x1, x2, [x11]                   // encoding: [0x61,0x11,0x62,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrpl  x21, x22, [sp]
// CHECK: ldclrpl x21, x22, [sp]                  // encoding: [0xf5,0x13,0x76,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldclrpl  x22, xzr, [sp]
// ERROR-INVALID-OP: [[@LINE-1]]:15: error: invalid operand for instruction
// ERROR-NO-LSE128: error: invalid operand for instruction
ldclrpl  xzr, x22, [sp]
// ERROR-INVALID-OP: [[@LINE-1]]:10: error: invalid operand for instruction
// ERROR-NO-LSE128: error: invalid operand for instruction

ldsetp   x1, x2, [x11]
// CHECK: ldsetp x1, x2, [x11]                   // encoding: [0x61,0x31,0x22,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetp   x21, x22, [sp]
// CHECK: ldsetp x21, x22, [sp]                  // encoding: [0xf5,0x33,0x36,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetpa  x1, x2, [x11]
// CHECK: ldsetpa x1, x2, [x11]                   // encoding: [0x61,0x31,0xa2,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetpa  x21, x22, [sp]
// CHECK: ldsetpa x21, x22, [sp]                  // encoding: [0xf5,0x33,0xb6,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetpal x1, x2, [x11]
// CHECK: ldsetpal x1, x2, [x11]                   // encoding: [0x61,0x31,0xe2,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetpal x21, x22, [sp]
// CHECK: ldsetpal x21, x22, [sp]                  // encoding: [0xf5,0x33,0xf6,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetpl  x1, x2, [x11]
// CHECK: ldsetpl x1, x2, [x11]                   // encoding: [0x61,0x31,0x62,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetpl  x21, x22, [sp]
// CHECK: ldsetpl x21, x22, [sp]                  // encoding: [0xf5,0x33,0x76,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
ldsetpl  x22, xzr, [sp]
// ERROR-INVALID-OP: [[@LINE-1]]:15: error: invalid operand for instruction
// ERROR-NO-LSE128: error: invalid operand for instruction
ldsetpl  xzr, x22, [sp]
// ERROR-INVALID-OP: [[@LINE-1]]:10: error: invalid operand for instruction
// ERROR-NO-LSE128: error: invalid operand for instruction


swpp     x1, x2, [x11]
// CHECK: swpp x1, x2, [x11]                   // encoding: [0x61,0x81,0x22,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swpp     x21, x22, [sp]
// CHECK: swpp x21, x22, [sp]                  // encoding: [0xf5,0x83,0x36,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swppa    x1, x2, [x11]
// CHECK: swppa x1, x2, [x11]                   // encoding: [0x61,0x81,0xa2,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swppa    x21, x22, [sp]
// CHECK: swppa x21, x22, [sp]                  // encoding: [0xf5,0x83,0xb6,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swppal   x1, x2, [x11]
// CHECK: swppal x1, x2, [x11]                   // encoding: [0x61,0x81,0xe2,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swppal   x21, x22, [sp]
// CHECK: swppal x21, x22, [sp]                  // encoding: [0xf5,0x83,0xf6,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swppl    x1, x2, [x11]
// CHECK: swppl x1, x2, [x11]                   // encoding: [0x61,0x81,0x62,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swppl    x21, x22, [sp]
// CHECK: swppl x21, x22, [sp]                  // encoding: [0xf5,0x83,0x76,0x19]
// ERROR-NO-LSE128: [[@LINE-2]]:1: error: instruction requires: lse128
swppl    x22, xzr, [sp]
// ERROR-INVALID-OP: [[@LINE-1]]:15: error: invalid operand for instruction
// ERROR-NO-LSE128: error: invalid operand for instruction
swppl    xzr, x22, [sp]
// ERROR-INVALID-OP: [[@LINE-1]]:10: error: invalid operand for instruction
// ERROR-NO-LSE128: error: invalid operand for instruction

