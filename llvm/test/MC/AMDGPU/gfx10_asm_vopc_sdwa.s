// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX10,W32 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX10,W64 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=GFX10-ERR,W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=GFX10-ERR,W64-ERR --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// ENC_VOPC, SDWA variant.
//===----------------------------------------------------------------------===//

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x01,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x01,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x03,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x03,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x05,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x05,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x07,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x07,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x09,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x09,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x0b,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x0b,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x0d,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x0d,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x0f,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x0f,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x11,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x11,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x13,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x13,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x15,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x15,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x17,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x17,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x19,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x19,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x1b,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x1b,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x1d,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x1d,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x1f,0x7c,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x1f,0x7c,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1e,0x7c,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x01,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x03,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x05,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x07,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x09,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x0b,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x0d,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_i32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x0f,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x01,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x00,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x03,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x02,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x05,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x04,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x07,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x06,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x09,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x08,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x0b,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0a,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x0d,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0c,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x0f,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x0e,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x11,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x11,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x10,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x13,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x15,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_i16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x17,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x19,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x1b,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x1d,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x13,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x12,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x15,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x14,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x17,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x16,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x19,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x18,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x1b,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1a,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x1d,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x1c,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_class_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x1f,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x53,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x55,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x57,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x59,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x5b,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x5d,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x81,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x83,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x85,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x87,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x89,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ne_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x8b,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x8d,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_t_u32_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x8f,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x0e,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s[6:7], v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x0e]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_f_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x91,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x93,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x95,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_le_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x97,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x99,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x9b,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x9d,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_o_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0x9f,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_u_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xd1,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xd3,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xd5,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xd7,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xd9,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xdb,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xdd,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[8:9], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x88,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[100:101], v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0xe4,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x00,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmp_tru_f16_sdwa s[6:7], v255, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0xff,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], s1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], s101, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x65,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x6a,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x6b,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], m0, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x7c,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x7e,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x7f,0x86,0x86,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v255 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0xfe,0xdf,0x7d,0x01,0x86,0x06,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x00,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x01,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x02,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x03,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x04,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x05,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x16,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x26,0x06]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x01]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x02]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x03]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x04]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x16]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s[6:7], v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W64: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x26]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x53,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x52,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x55,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x54,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x57,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x56,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x59,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x58,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x5b,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5a,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x5d,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x5c,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x81,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x80,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x83,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x82,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x85,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x84,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x87,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x86,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x89,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x88,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x8b,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8a,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x8d,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8c,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x8f,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x0e,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x86,0x06,0x0e]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x91,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x90,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x93,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x92,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x95,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x94,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x97,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x96,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x99,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x98,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x9b,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9a,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x9d,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9c,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0x9f,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0x9e,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xd1,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd0,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xd3,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd2,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xd5,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd4,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xd7,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd6,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xd9,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xd8,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xdb,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xda,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xdd,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xdc,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s8, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x88,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0xe4,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x00,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v255, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0xff,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, s101, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x65,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, vcc_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x6a,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, vcc_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x6b,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x7c,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, exec_lo, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x7e,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x7f,0x86,0x86,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0xfe,0xdf,0x7d,0x01,0x86,0x06,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x00,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x01,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x02,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x03,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x04,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x05,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x16,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x26,0x06]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_1
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x01]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_2
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x02]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_3
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x03]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_0
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x04]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x05]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x16]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD
// W32: encoding: [0xf9,0x04,0xde,0x7d,0x01,0x86,0x06,0x26]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
