// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=W32 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=W64 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W64-ERR --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// ENC_VOPC - v_cmp_* opcodes.
//===----------------------------------------------------------------------===//

v_cmp_f_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x00,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x00,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x00,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x01,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x00,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x00,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x00,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x01,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x02,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x02,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x02,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x03,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x02,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x02,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x02,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x03,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x04,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x04,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x04,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x05,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x04,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x04,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x04,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x05,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x06,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x06,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x06,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x07,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x06,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x06,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x06,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x07,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x08,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x08,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x08,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x09,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x08,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x08,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x08,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x09,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x0a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x0a,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x0a,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x0b,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x0a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x0a,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x0a,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x0b,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x0c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x0c,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x0c,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x0d,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x0c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x0c,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x0c,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x0d,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x0e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x0e,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x0e,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x0f,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x0e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x0e,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x0e,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x0f,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x10,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x10,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x10,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x11,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x10,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x10,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x10,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x11,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x12,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x12,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x12,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x13,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x12,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x12,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x12,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x13,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x14,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x14,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x14,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x15,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x14,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x14,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x14,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x15,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x16,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x16,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x16,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x17,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x16,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x16,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x16,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x17,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x18,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x18,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x18,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x19,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x18,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x18,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x18,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x19,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x1a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x1a,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x1a,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x1b,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x1a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x1a,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x1a,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x1b,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x1c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x1c,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x1c,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x1d,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x1c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x1c,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x1c,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x1d,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x1e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x1e,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x1e,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x1f,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x1e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x1e,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x1e,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x1f,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x40,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x40,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x40,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x41,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x40,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x40,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x40,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x41,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x42,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x42,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x43,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x42,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x42,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x43,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x44,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x44,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x44,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x45,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x44,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x44,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x44,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x45,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x46,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x46,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x46,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x47,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x46,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x46,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x46,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x47,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x48,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x48,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x48,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x49,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x48,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x48,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x48,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x49,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x4a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x4a,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x4a,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x4b,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x4a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x4a,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x4a,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x4b,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x4c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x4c,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x4c,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x4d,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x4c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x4c,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x4c,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x4d,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x4e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x4e,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x4e,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x4f,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x4e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x4e,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x4e,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x4f,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x50,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x50,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x50,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x51,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x50,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x50,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x50,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x51,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x52,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x52,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x52,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x53,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x52,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x52,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x52,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x53,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x54,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x54,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x54,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x55,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x54,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x54,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x54,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x55,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x56,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x56,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x56,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x57,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x56,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x56,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x56,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x57,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x58,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x58,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x58,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x59,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x58,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x58,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x58,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x59,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x5a,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x5a,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x5a,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x5b,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x5a,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x5a,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x5a,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x5b,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x5c,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x5c,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x5c,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x5d,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x5c,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x5c,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x5c,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x5d,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x5e,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x5e,0x7c,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x5e,0x7c,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x5f,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x5e,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x5e,0x7c,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x5e,0x7c,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x5f,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x00,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x00,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x00,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x01,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x02,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x02,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x02,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x03,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x04,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x04,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x04,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x05,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x06,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x06,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x06,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x07,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x08,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x08,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x08,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x09,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x0a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x0a,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x0a,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x0b,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x0c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x0c,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x0c,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x0d,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x0e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x0e,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x0e,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x0f,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x00,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x00,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x00,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x01,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x02,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x02,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x02,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x03,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x04,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x04,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x04,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x05,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x06,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x06,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x06,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x07,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x08,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x08,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x08,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x09,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x0a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x0a,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x0a,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x0b,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x0c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x0c,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x0c,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x0d,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x0e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x0e,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x0e,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x0f,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x10,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x10,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x10,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x11,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x10,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x10,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x10,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x11,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x12,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x12,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x12,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x12,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x12,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x13,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x14,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x14,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x14,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x14,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x14,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x15,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x16,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x16,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x16,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x16,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x16,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x17,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x18,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x18,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x18,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x18,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x18,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x19,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x1a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x1a,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x1a,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x1a,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x1a,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x1b,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x1c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x1c,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x1c,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x1c,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x1c,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x1d,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x12,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x12,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x12,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x12,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x12,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x13,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x14,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x14,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x14,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x14,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x14,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x15,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x16,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x16,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x16,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x16,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x16,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x17,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x18,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x18,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x18,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x18,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x18,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x19,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x1a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x1a,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x1a,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x1a,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x1a,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x1b,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x1c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x1c,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x1c,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x1c,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x1c,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x1d,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x1e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x1e,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x1e,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x1f,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x40,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x40,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x40,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x41,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x42,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x42,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x42,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x43,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x44,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x44,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x44,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x45,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x46,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x46,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x46,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x47,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x48,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x48,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x48,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x49,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x4a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x4a,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x4a,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x4b,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x4c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x4c,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x4c,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x4d,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0x4e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0x4e,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0x4e,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0x4f,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, v[1:2], v2
// W64: encoding: [0x01,0x05,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, v[254:255], v2
// W64: encoding: [0xfe,0x05,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, s[2:3], v2
// W64: encoding: [0x02,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, s[4:5], v2
// W64: encoding: [0x04,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, s[100:101], v2
// W64: encoding: [0x64,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, vcc, v2
// W64: encoding: [0x6a,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, exec, v2
// W64: encoding: [0x7e,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x50,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x50,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x50,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc, v[1:2], v255
// W64: encoding: [0x01,0xff,0x51,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xc0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xc0,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xc0,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xc1,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xc2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xc2,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xc2,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xc3,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xc4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xc4,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xc4,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xc5,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xc6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xc6,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xc6,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xc7,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xc8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xc8,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xc8,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xc9,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xca,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xca,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xca,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xcb,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xcc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xcc,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xcc,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xcd,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x01,0x05,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, v[254:255], v[2:3]
// W64: encoding: [0xfe,0x05,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, s[2:3], v[2:3]
// W64: encoding: [0x02,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, s[4:5], v[2:3]
// W64: encoding: [0x04,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, s[100:101], v[2:3]
// W64: encoding: [0x64,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, vcc, v[2:3]
// W64: encoding: [0x6a,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, exec, v[2:3]
// W64: encoding: [0x7e,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, 0, v[2:3]
// W64: encoding: [0x80,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, -1, v[2:3]
// W64: encoding: [0xc1,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, 0.5, v[2:3]
// W64: encoding: [0xf0,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, -4.0, v[2:3]
// W64: encoding: [0xf7,0x04,0xce,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, 0xaf123456, v[2:3]
// W64: encoding: [0xff,0x04,0xce,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, 0x3f717273, v[2:3]
// W64: encoding: [0xff,0x04,0xce,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc, v[1:2], v[254:255]
// W64: encoding: [0x01,0xfd,0xcf,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x40,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x40,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x40,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x41,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x42,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x42,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x42,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x43,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x44,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x44,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x44,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x45,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x46,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x46,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x46,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x47,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x48,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x48,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x48,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x49,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x4a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x4a,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x4a,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x4b,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x4c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x4c,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x4c,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x4d,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0x4e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0x4e,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0x4e,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0x4f,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, v[1:2], v2
// W32: encoding: [0x01,0x05,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, v[254:255], v2
// W32: encoding: [0xfe,0x05,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, s[2:3], v2
// W32: encoding: [0x02,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, s[4:5], v2
// W32: encoding: [0x04,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, s[100:101], v2
// W32: encoding: [0x64,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, vcc, v2
// W32: encoding: [0x6a,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, exec, v2
// W32: encoding: [0x7e,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x50,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x50,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x50,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f64 vcc_lo, v[1:2], v255
// W32: encoding: [0x01,0xff,0x51,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xc0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xc0,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xc0,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xc1,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xc2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xc2,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xc2,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xc3,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xc4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xc4,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xc4,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xc5,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xc6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xc6,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xc6,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xc7,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xc8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xc8,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xc8,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xc9,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xca,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xca,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xca,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xcb,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xcc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xcc,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xcc,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xcd,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x01,0x05,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, v[254:255], v[2:3]
// W32: encoding: [0xfe,0x05,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, s[2:3], v[2:3]
// W32: encoding: [0x02,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, s[4:5], v[2:3]
// W32: encoding: [0x04,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, s[100:101], v[2:3]
// W32: encoding: [0x64,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, vcc, v[2:3]
// W32: encoding: [0x6a,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, exec, v[2:3]
// W32: encoding: [0x7e,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, 0, v[2:3]
// W32: encoding: [0x80,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, -1, v[2:3]
// W32: encoding: [0xc1,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, 0.5, v[2:3]
// W32: encoding: [0xf0,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, -4.0, v[2:3]
// W32: encoding: [0xf7,0x04,0xce,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, 0xaf123456, v[2:3]
// W32: encoding: [0xff,0x04,0xce,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, 0x3f717273, v[2:3]
// W32: encoding: [0xff,0x04,0xce,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u64 vcc_lo, v[1:2], v[254:255]
// W32: encoding: [0x01,0xfd,0xcf,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x52,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x52,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x52,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x52,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x52,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x53,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x54,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x54,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x54,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x54,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x54,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x55,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x56,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x56,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x56,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x56,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x56,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x57,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x58,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x58,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x58,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x58,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x58,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x59,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x5a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x5a,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x5a,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x5a,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x5a,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x5b,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x5c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, 0.5, v2
// W64: encoding: [0xff,0x04,0x5c,0x7d,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, -4.0, v2
// W64: encoding: [0xff,0x04,0x5c,0x7d,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x5c,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x5c,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x5d,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x80,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x80,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x80,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x81,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x82,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x82,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x82,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x83,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x84,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x84,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x84,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x85,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x86,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x86,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x86,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x87,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x88,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x88,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x88,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x89,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x8a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x8a,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x8a,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x8b,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x8c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x8c,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x8c,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x8d,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x8e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, 0xaf123456, v2
// W64: encoding: [0xff,0x04,0x8e,0x7d,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, 0x3f717273, v2
// W64: encoding: [0xff,0x04,0x8e,0x7d,0x73,0x72,0x71,0x3f]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x8f,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x90,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x90,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x90,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x91,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x92,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x92,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x92,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x93,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x94,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x94,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x94,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x95,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x96,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x96,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x96,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x97,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x98,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x98,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x98,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x99,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x9a,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x9a,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x9a,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x9b,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x9c,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x9c,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x9c,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x9d,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0x9e,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0x9e,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0x9e,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0x9f,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xd0,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xd0,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xd0,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xd1,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xd2,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xd2,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xd2,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xd3,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xd4,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xd4,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xd4,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xd5,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xd6,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xd6,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xd6,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xd7,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xd8,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xd8,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xd8,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xd9,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xda,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xda,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xda,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xdb,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xdc,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xdc,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xdc,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xdd,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, v1, v2
// W64: encoding: [0x01,0x05,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, v255, v2
// W64: encoding: [0xff,0x05,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, s1, v2
// W64: encoding: [0x01,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, s101, v2
// W64: encoding: [0x65,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, vcc_lo, v2
// W64: encoding: [0x6a,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, vcc_hi, v2
// W64: encoding: [0x6b,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, m0, v2
// W64: encoding: [0x7c,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, exec_lo, v2
// W64: encoding: [0x7e,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, exec_hi, v2
// W64: encoding: [0x7f,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, 0, v2
// W64: encoding: [0x80,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, -1, v2
// W64: encoding: [0xc1,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, 0.5, v2
// W64: encoding: [0xf0,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, -4.0, v2
// W64: encoding: [0xf7,0x04,0xde,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, 0xfe0b, v2
// W64: encoding: [0xff,0x04,0xde,0x7d,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, 0x3456, v2
// W64: encoding: [0xff,0x04,0xde,0x7d,0x56,0x34,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc, v1, v255
// W64: encoding: [0x01,0xff,0xdf,0x7d]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x52,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x52,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x52,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x52,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x52,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x53,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x54,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x54,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x54,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x54,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x54,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x55,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x56,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x56,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x56,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x56,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x56,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x57,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x58,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x58,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x58,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x58,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x58,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x59,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x5a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x5a,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x5a,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x5a,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x5a,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x5b,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x5c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, 0.5, v2
// W32: encoding: [0xff,0x04,0x5c,0x7d,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, -4.0, v2
// W32: encoding: [0xff,0x04,0x5c,0x7d,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x5c,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x5c,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x5d,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x80,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x80,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x80,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x81,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x82,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x82,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x82,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x83,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x84,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x84,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x84,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x85,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x86,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x86,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x86,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x87,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x88,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x88,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x88,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x89,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x8a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x8a,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x8a,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x8b,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x8c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x8c,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x8c,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x8d,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x8e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, 0xaf123456, v2
// W32: encoding: [0xff,0x04,0x8e,0x7d,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, 0x3f717273, v2
// W32: encoding: [0xff,0x04,0x8e,0x7d,0x73,0x72,0x71,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x8f,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x90,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x90,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x90,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x91,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x92,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x92,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x92,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x93,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x94,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x94,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x94,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x95,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x96,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x96,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x96,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x97,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x98,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x98,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x98,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x99,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x9a,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x9a,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x9a,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x9b,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x9c,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x9c,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x9c,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x9d,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0x9e,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0x9e,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0x9e,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0x9f,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xd0,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xd0,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xd0,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xd1,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xd2,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xd2,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xd2,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xd3,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xd4,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xd4,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xd4,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xd5,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xd6,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xd6,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xd6,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xd7,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xd8,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xd8,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xd8,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xd9,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xda,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xda,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xda,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xdb,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xdc,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xdc,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xdc,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xdd,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, v1, v2
// W32: encoding: [0x01,0x05,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, v255, v2
// W32: encoding: [0xff,0x05,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, s1, v2
// W32: encoding: [0x01,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, s101, v2
// W32: encoding: [0x65,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, vcc_lo, v2
// W32: encoding: [0x6a,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, vcc_hi, v2
// W32: encoding: [0x6b,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, m0, v2
// W32: encoding: [0x7c,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, exec_lo, v2
// W32: encoding: [0x7e,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, exec_hi, v2
// W32: encoding: [0x7f,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, 0, v2
// W32: encoding: [0x80,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, -1, v2
// W32: encoding: [0xc1,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, 0.5, v2
// W32: encoding: [0xf0,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, -4.0, v2
// W32: encoding: [0xf7,0x04,0xde,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, 0xfe0b, v2
// W32: encoding: [0xff,0x04,0xde,0x7d,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, 0x3456, v2
// W32: encoding: [0xff,0x04,0xde,0x7d,0x56,0x34,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16 vcc_lo, v1, v255
// W32: encoding: [0x01,0xff,0xdf,0x7d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
