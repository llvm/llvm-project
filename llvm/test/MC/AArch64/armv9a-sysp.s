// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// sysp #<op1>, <Cn>, <Cm>, #<op2>{, <Xt1>, <Xt2>}
// registers with 128-bit formats (op0, op1, Cn, Cm, op2)
// For sysp, op0 is 0

sysp #0, c8, c0, #0, x0, x1
// CHECK-INST: sysp #0, c8, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x80,0x48,0xd5]

sysp #0, c8, c0, #1, x0, x1
// CHECK-INST: sysp #0, c8, c0, #1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x80,0x48,0xd5]

sysp #0, c8, c4, #0, x0, x1
// CHECK-INST: sysp #0, c8, c4, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x84,0x48,0xd5]

sysp #0, c8, c0, #3, x0, x1
// CHECK-INST: sysp #0, c8, c0, #3, x0, x1
// CHECK-ENCODING: encoding: [0x60,0x80,0x48,0xd5]

sysp #0, c8, c0, #6, x0, x1
// CHECK-INST: sysp #0, c8, c0, #6, x0, x1
// CHECK-ENCODING: encoding: [0xc0,0x80,0x48,0xd5]

sysp #4, c8, c0, #0, x0, x1
// CHECK-INST: sysp #4, c8, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x80,0x4c,0xd5]

sysp #4, c8, c0, #3, x0, x1
// CHECK-INST: sysp #4, c8, c0, #3, x0, x1
// CHECK-ENCODING: encoding: [0x60,0x80,0x4c,0xd5]

sysp #4, c8, c1, #0, x0, x1
// CHECK-INST: sysp #4, c8, c1, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x81,0x4c,0xd5]

sysp #0, c8, c0, #0, x0, x1
// CHECK-INST: sysp #0, c8, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x80,0x48,0xd5]

sysp #0, c8, c0, #1, x0, x1
// CHECK-INST: sysp #0, c8, c0, #1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x80,0x48,0xd5]

sysp #0, c8, c4, #0, x0, x1
// CHECK-INST: sysp #0, c8, c4, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x84,0x48,0xd5]

sysp #0, c8, c0, #3, x0, x1
// CHECK-INST: sysp #0, c8, c0, #3, x0, x1
// CHECK-ENCODING: encoding: [0x60,0x80,0x48,0xd5]

sysp #0, c8, c0, #6, x0, x1
// CHECK-INST: sysp #0, c8, c0, #6, x0, x1
// CHECK-ENCODING: encoding: [0xc0,0x80,0x48,0xd5]

sysp #4, c8, c0, #0, x0, x1
// CHECK-INST: sysp #4, c8, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x80,0x4c,0xd5]

sysp #4, c8, c0, #3, x0, x1
// CHECK-INST: sysp #4, c8, c0, #3, x0, x1
// CHECK-ENCODING: encoding: [0x60,0x80,0x4c,0xd5]

sysp #4, c8, c1, #0, x0, x1
// CHECK-INST: sysp #4, c8, c1, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x81,0x4c,0xd5]

sysp #0, c8, c0, #0, x0, x1
// CHECK-INST: sysp #0, c8, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x2, x3
// CHECK-INST: sysp #0, c8, c0, #0, x2, x3
// CHECK-ENCODING: encoding: [0x02,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x4, x5
// CHECK-INST: sysp #0, c8, c0, #0, x4, x5
// CHECK-ENCODING: encoding: [0x04,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x6, x7
// CHECK-INST: sysp #0, c8, c0, #0, x6, x7
// CHECK-ENCODING: encoding: [0x06,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x8, x9
// CHECK-INST: sysp #0, c8, c0, #0, x8, x9
// CHECK-ENCODING: encoding: [0x08,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x10, x11
// CHECK-INST: sysp #0, c8, c0, #0, x10, x11
// CHECK-ENCODING: encoding: [0x0a,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x12, x13
// CHECK-INST: sysp #0, c8, c0, #0, x12, x13
// CHECK-ENCODING: encoding: [0x0c,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x14, x15
// CHECK-INST: sysp #0, c8, c0, #0, x14, x15
// CHECK-ENCODING: encoding: [0x0e,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x16, x17
// CHECK-INST: sysp #0, c8, c0, #0, x16, x17
// CHECK-ENCODING: encoding: [0x10,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x18, x19
// CHECK-INST: sysp #0, c8, c0, #0, x18, x19
// CHECK-ENCODING: encoding: [0x12,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x20, x21
// CHECK-INST: sysp #0, c8, c0, #0, x20, x21
// CHECK-ENCODING: encoding: [0x14,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x22, x23
// CHECK-INST: sysp #0, c8, c0, #0, x22, x23
// CHECK-ENCODING: encoding: [0x16,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x24, x25
// CHECK-INST: sysp #0, c8, c0, #0, x24, x25
// CHECK-ENCODING: encoding: [0x18,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x26, x27
// CHECK-INST: sysp #0, c8, c0, #0, x26, x27
// CHECK-ENCODING: encoding: [0x1a,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x28, x29
// CHECK-INST: sysp #0, c8, c0, #0, x28, x29
// CHECK-ENCODING: encoding: [0x1c,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x30, x31
// CHECK-INST: sysp #0, c8, c0, #0, x30, xzr
// CHECK-ENCODING: encoding: [0x1e,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x31, x31
// CHECK-INST: sysp #0, c8, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, xzr, xzr
// CHECK-INST: sysp #0, c8, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, x31, xzr
// CHECK-INST: sysp #0, c8, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x80,0x48,0xd5]

sysp #0, c8, c0, #0, xzr, x31
// CHECK-INST: sysp #0, c8, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x80,0x48,0xd5]

sysp #0, c8, c0, #0
// CHECK-INST: sysp #0, c8, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x80,0x48,0xd5]
