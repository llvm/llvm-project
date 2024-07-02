// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX12,W32 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX12,W64 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W64-ERR --implicit-check-not=error: %s

v_cmp_class_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, v255, v2
// W32: encoding: [0x05,0x00,0x7d,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, s1, v2
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, s105, v255
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x69,0xfe,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, vcc_lo, s2
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x6a,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, vcc_hi, s105
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x6b,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, ttmp15, ttmp15
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x7b,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, m0, src_scc
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x7d,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x7d,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x7d,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x7d,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x7d,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x7d,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], s105, v255
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x69,0xfe,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], vcc_lo, s2
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x6a,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], vcc_hi, s105
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x6b,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], ttmp15, ttmp15
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x7b,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], m0, src_scc
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x7d,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x7d,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x7d,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x7d,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x7d,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 null, -|0xfe0b|, vcc_hi
// GFX12: encoding: [0x7c,0x01,0x7d,0xd4,0xff,0xd6,0x00,0x20,0x0b,0xfe,0x00,0x00]

v_cmp_class_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x7e,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x7e,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x7e,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x7e,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x7e,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x7e,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x7e,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x7e,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x7e,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x7e,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 null, -|0xaf123456|, vcc_hi
// GFX12: encoding: [0x7c,0x01,0x7e,0xd4,0xff,0xd6,0x00,0x20,0x56,0x34,0x12,0xaf]

v_cmp_class_f64_e64 s5, v[1:2], v2
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, v[1:2], v255
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, v[1:2], s2
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, v[1:2], s105
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x01,0xd3,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, v[254:255], ttmp15
// W32: encoding: [0x05,0x00,0x7f,0xd4,0xfe,0xf7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, s[2:3], vcc_hi
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x02,0xd6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, s[104:105], vcc_lo
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x68,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, vcc, m0
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x6a,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, ttmp[14:15], exec_hi
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x7a,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s5, exec, exec_lo
// W32: encoding: [0x05,0x00,0x7f,0xd4,0x7e,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s105, null, null
// W32: encoding: [0x69,0x00,0x7f,0xd4,0x7c,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x7f,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 vcc_hi, 0.5, 0.5
// W32: encoding: [0x6b,0x00,0x7f,0xd4,0xf0,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 ttmp15, -|src_scc|, src_scc
// W32: encoding: [0x7b,0x01,0x7f,0xd4,0xfd,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], v2
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], v255
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], s2
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], s105
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x01,0xd3,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[254:255], ttmp15
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0xfe,0xf7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], s[2:3], vcc_hi
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x02,0xd6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], s[104:105], vcc_lo
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x68,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], vcc, m0
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x6a,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], ttmp[14:15], exec_hi
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x7a,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], exec, exec_lo
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x7e,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], null, null
// W64: encoding: [0x0a,0x00,0x7f,0xd4,0x7c,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x7f,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 vcc, 0.5, 0.5
// W64: encoding: [0x6a,0x00,0x7f,0xd4,0xf0,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 ttmp[14:15], -|src_scc|, src_scc
// W64: encoding: [0x7a,0x01,0x7f,0xd4,0xfd,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 null, 0xaf123456, 0xaf123456
// GFX12: encoding: [0x7c,0x00,0x7f,0xd4,0xff,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]

v_cmp_eq_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x02,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x02,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x02,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x02,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x02,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x02,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x02,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x02,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x02,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x02,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x02,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x02,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x02,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x02,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x02,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x02,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x02,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x02,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x02,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_eq_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x12,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x12,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x12,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x12,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x12,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x12,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x12,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x12,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x12,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x12,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x12,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x12,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x12,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x12,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x12,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x12,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x12,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x12,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x12,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x12,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x12,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_eq_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x22,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x22,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x22,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x22,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x22,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x22,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x22,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x22,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x22,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x22,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x22,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x22,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x22,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x22,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x22,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_eq_i16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x32,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x32,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x32,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x32,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x32,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x32,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x32,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x32,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x32,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x32,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x32,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x32,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x32,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x32,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x32,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x32,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x32,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x32,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x32,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x32,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_eq_i32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x42,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x42,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x42,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x42,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x42,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x42,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x42,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x42,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x42,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x42,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x42,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x42,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x42,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x42,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x42,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x42,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x42,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x42,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x42,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x42,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_eq_i64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x52,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x52,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x52,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x52,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x52,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x52,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x52,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x52,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x52,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x52,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x52,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x52,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x52,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x52,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x52,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x52,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x52,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x52,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x52,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x52,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x52,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x52,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x52,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_eq_u16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x3a,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x3a,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x3a,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x3a,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x3a,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x3a,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x3a,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x3a,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x3a,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x3a,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x3a,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_eq_u32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x4a,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x4a,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x4a,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x4a,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x4a,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x4a,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x4a,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x4a,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x4a,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x4a,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x4a,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_eq_u64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x5a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x5a,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x5a,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x5a,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x5a,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x5a,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x5a,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x5a,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x5a,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x5a,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x5a,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x5a,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x5a,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x5a,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x5a,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x5a,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ge_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x06,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x06,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x06,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x06,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x06,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x06,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x06,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x06,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x06,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x06,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x06,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x06,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x06,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x06,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x06,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x06,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x06,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x06,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x06,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_ge_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x16,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x16,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x16,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x16,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x16,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x16,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x16,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x16,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x16,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x16,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x16,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x16,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x16,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x16,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x16,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x16,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x16,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x16,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x16,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x16,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x16,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_ge_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x26,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x26,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x26,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x26,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x26,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x26,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x26,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x26,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x26,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x26,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x26,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x26,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x26,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x26,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x26,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_ge_i16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x36,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x36,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x36,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x36,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x36,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x36,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x36,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x36,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x36,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x36,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x36,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x36,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x36,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x36,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x36,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x36,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x36,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x36,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x36,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x36,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_ge_i32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x46,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x46,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x46,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x46,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x46,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x46,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x46,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x46,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x46,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x46,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x46,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x46,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x46,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x46,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x46,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x46,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x46,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x46,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x46,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x46,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ge_i64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x56,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x56,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x56,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x56,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x56,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x56,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x56,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x56,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x56,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x56,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x56,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x56,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x56,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x56,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x56,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x56,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x56,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x56,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x56,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x56,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x56,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x56,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x56,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ge_u16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x3e,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x3e,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x3e,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x3e,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x3e,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x3e,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x3e,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x3e,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x3e,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x3e,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x3e,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_ge_u32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x4e,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x4e,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x4e,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x4e,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x4e,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x4e,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x4e,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x4e,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x4e,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x4e,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x4e,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ge_u64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x5e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x5e,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x5e,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x5e,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x5e,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x5e,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x5e,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x5e,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x5e,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x5e,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x5e,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x5e,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x5e,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x5e,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x5e,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x5e,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_gt_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x04,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x04,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x04,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x04,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x04,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x04,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x04,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x04,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x04,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x04,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x04,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x04,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x04,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x04,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x04,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x04,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x04,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x04,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x04,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_gt_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x14,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x14,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x14,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x14,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x14,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x14,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x14,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x14,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x14,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x14,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x14,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x14,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x14,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x14,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x14,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x14,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x14,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x14,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x14,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x14,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x14,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_gt_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x24,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x24,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x24,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x24,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x24,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x24,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x24,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x24,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x24,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x24,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x24,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x24,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x24,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x24,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x24,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_gt_i16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x34,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x34,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x34,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x34,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x34,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x34,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x34,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x34,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x34,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x34,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x34,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x34,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x34,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x34,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x34,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x34,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x34,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x34,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x34,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x34,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_gt_i32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x44,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x44,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x44,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x44,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x44,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x44,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x44,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x44,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x44,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x44,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x44,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x44,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x44,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x44,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x44,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x44,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x44,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x44,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x44,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x44,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_gt_i64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x54,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x54,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x54,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x54,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x54,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x54,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x54,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x54,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x54,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x54,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x54,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x54,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x54,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x54,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x54,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x54,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x54,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x54,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x54,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x54,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x54,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x54,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x54,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_gt_u16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x3c,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x3c,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x3c,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x3c,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x3c,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x3c,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x3c,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x3c,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x3c,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x3c,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x3c,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_gt_u32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x4c,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x4c,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x4c,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x4c,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x4c,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x4c,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x4c,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x4c,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x4c,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x4c,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x4c,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_gt_u64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x5c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x5c,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x5c,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x5c,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x5c,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x5c,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x5c,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x5c,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x5c,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x5c,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x5c,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x5c,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x5c,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x5c,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x5c,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x5c,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_le_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x03,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x03,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x03,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x03,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x03,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x03,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x03,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x03,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x03,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x03,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x03,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x03,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x03,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x03,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x03,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x03,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x03,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x03,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x03,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_le_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x13,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x13,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x13,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x13,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x13,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x13,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x13,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x13,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x13,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x13,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x13,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x13,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x13,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x13,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x13,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x13,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x13,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x13,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x13,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x13,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x13,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_le_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x23,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x23,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x23,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x23,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x23,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x23,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x23,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x23,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x23,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x23,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x23,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x23,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x23,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x23,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x23,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_le_i16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x33,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x33,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x33,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x33,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x33,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x33,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x33,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x33,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x33,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x33,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x33,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x33,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x33,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x33,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x33,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x33,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x33,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x33,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x33,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x33,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_le_i32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x43,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x43,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x43,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x43,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x43,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x43,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x43,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x43,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x43,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x43,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x43,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x43,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x43,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x43,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x43,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x43,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x43,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x43,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x43,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x43,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_le_i64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x53,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x53,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x53,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x53,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x53,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x53,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x53,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x53,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x53,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x53,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x53,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x53,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x53,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x53,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x53,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x53,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x53,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x53,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x53,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x53,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x53,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x53,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x53,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_le_u16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x3b,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x3b,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x3b,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x3b,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x3b,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x3b,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x3b,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x3b,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x3b,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x3b,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x3b,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_le_u32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x4b,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x4b,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x4b,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x4b,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x4b,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x4b,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x4b,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x4b,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x4b,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x4b,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x4b,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_le_u64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x5b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x5b,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x5b,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x5b,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x5b,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x5b,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x5b,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x5b,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x5b,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x5b,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x5b,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x5b,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x5b,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x5b,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x5b,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x5b,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_lg_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x05,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x05,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x05,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x05,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x05,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x05,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x05,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x05,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x05,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x05,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x05,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x05,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x05,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x05,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x05,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x05,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x05,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x05,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x05,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_lg_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x15,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x15,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x15,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x15,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x15,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x15,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x15,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x15,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x15,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x15,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x15,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x15,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x15,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x15,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x15,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x15,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x15,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x15,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x15,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x15,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x15,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_lg_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x25,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x25,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x25,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x25,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x25,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x25,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x25,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x25,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x25,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x25,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x25,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x25,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x25,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x25,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x25,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_lt_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x01,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x01,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x01,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x01,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x01,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x01,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x01,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x01,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x01,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x01,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x01,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x01,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x01,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x01,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x01,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x01,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x01,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x01,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x01,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_lt_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x11,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x11,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x11,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x11,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x11,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x11,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x11,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x11,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x11,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x11,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x11,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x11,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x11,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x11,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x11,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x11,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x11,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x11,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x11,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x11,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x11,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_lt_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x21,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x21,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x21,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x21,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x21,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x21,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x21,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x21,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x21,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x21,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x21,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x21,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x21,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x21,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x21,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_lt_i16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x31,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x31,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x31,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x31,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x31,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x31,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x31,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x31,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x31,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x31,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x31,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x31,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x31,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x31,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x31,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x31,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x31,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x31,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x31,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x31,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_lt_i32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x41,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x41,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x41,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x41,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x41,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x41,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x41,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x41,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x41,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x41,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x41,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x41,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x41,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x41,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x41,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x41,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x41,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x41,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x41,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x41,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_lt_i64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x51,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x51,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x51,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x51,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x51,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x51,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x51,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x51,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x51,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x51,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x51,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x51,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x51,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x51,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x51,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x51,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x51,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x51,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x51,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x51,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x51,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x51,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x51,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_lt_u16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x39,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x39,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x39,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x39,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x39,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x39,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x39,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x39,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x39,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x39,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x39,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x39,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x39,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x39,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x39,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x39,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x39,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x39,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x39,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x39,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_lt_u32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x49,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x49,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x49,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x49,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x49,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x49,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x49,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x49,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x49,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x49,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x49,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x49,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x49,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x49,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x49,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x49,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x49,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x49,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x49,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x49,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_lt_u64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x59,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x59,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x59,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x59,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x59,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x59,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x59,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x59,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x59,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x59,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x59,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x59,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x59,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x59,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x59,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x59,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x59,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x59,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x59,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x59,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x59,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x59,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x59,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ne_i16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x35,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x35,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x35,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x35,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x35,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x35,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x35,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x35,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x35,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x35,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x35,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x35,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x35,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x35,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x35,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x35,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x35,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x35,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x35,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x35,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_ne_i32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x45,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x45,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x45,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x45,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x45,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x45,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x45,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x45,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x45,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x45,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x45,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x45,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x45,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x45,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x45,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x45,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x45,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x45,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x45,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x45,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ne_i64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x55,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x55,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x55,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x55,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x55,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x55,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x55,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x55,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x55,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x55,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x55,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x55,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x55,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x55,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x55,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x55,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x55,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x55,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x55,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x55,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x55,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x55,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x55,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ne_u16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x3d,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x3d,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x3d,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x3d,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x3d,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x3d,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x3d,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x3d,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x3d,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x3d,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 null, 0xfe0b, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x3d,0xd4,0xff,0xd6,0x00,0x00,0x0b,0xfe,0x00,0x00]

v_cmp_ne_u32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x4d,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s5, exec_hi, null
// W32: encoding: [0x05,0x00,0x4d,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x4d,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x4d,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 vcc_hi, 0.5, m0
// W32: encoding: [0x6b,0x00,0x4d,0xd4,0xf0,0xfa,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 ttmp15, src_scc, vcc_lo
// W32: encoding: [0x7b,0x00,0x4d,0xd4,0xfd,0xd4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], exec_hi, null
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x4d,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x4d,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 vcc, 0.5, m0
// W64: encoding: [0x6a,0x00,0x4d,0xd4,0xf0,0xfa,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 ttmp[14:15], src_scc, vcc_lo
// W64: encoding: [0x7a,0x00,0x4d,0xd4,0xfd,0xd4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 null, 0xaf123456, vcc_hi
// GFX12: encoding: [0x7c,0x00,0x4d,0xd4,0xff,0xd6,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_ne_u64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x5d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x5d,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x5d,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x5d,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x5d,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x5d,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s5, exec, src_scc
// W32: encoding: [0x05,0x00,0x5d,0xd4,0x7e,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x5d,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x5d,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x5d,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 ttmp15, src_scc, exec
// W32: encoding: [0x7b,0x00,0x5d,0xd4,0xfd,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], exec, src_scc
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0x7e,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x5d,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x5d,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x5d,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 ttmp[14:15], src_scc, exec
// W64: encoding: [0x7a,0x00,0x5d,0xd4,0xfd,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 null, 0xaf123456, vcc
// GFX12: encoding: [0x7c,0x00,0x5d,0xd4,0xff,0xd4,0x00,0x00,0x56,0x34,0x12,0xaf]

v_cmp_neq_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x0d,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x0d,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x0d,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x0d,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x0d,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x0d,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x0d,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x0d,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x0d,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x0d,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x0d,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x0d,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_neq_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x1d,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x1d,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x1d,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x1d,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x1d,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x1d,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x1d,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x1d,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x1d,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x1d,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x1d,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x1d,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x1d,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_neq_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x2d,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x2d,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x2d,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x2d,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x2d,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x2d,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x2d,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x2d,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x2d,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x2d,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x2d,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x2d,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x2d,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x2d,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x2d,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_nge_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x09,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x09,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x09,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x09,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x09,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x09,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x09,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x09,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x09,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x09,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x09,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x09,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x09,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x09,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x09,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x09,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x09,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x09,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x09,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_nge_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x19,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x19,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x19,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x19,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x19,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x19,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x19,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x19,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x19,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x19,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x19,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x19,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x19,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x19,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x19,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x19,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x19,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x19,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x19,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x19,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x19,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_nge_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x29,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x29,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x29,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x29,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x29,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x29,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x29,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x29,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x29,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x29,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x29,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x29,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x29,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x29,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x29,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_ngt_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x0b,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x0b,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x0b,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x0b,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x0b,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x0b,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x0b,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x0b,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x0b,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x0b,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x0b,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x0b,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_ngt_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x1b,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x1b,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x1b,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x1b,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x1b,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x1b,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x1b,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x1b,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x1b,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x1b,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x1b,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x1b,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x1b,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_ngt_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x2b,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x2b,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x2b,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x2b,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x2b,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x2b,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x2b,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x2b,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x2b,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x2b,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x2b,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x2b,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x2b,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x2b,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x2b,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_nle_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x0c,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x0c,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x0c,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x0c,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x0c,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x0c,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x0c,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x0c,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x0c,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x0c,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x0c,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x0c,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_nle_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x1c,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x1c,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x1c,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x1c,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x1c,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x1c,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x1c,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x1c,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x1c,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x1c,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x1c,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x1c,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x1c,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_nle_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x2c,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x2c,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x2c,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x2c,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x2c,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x2c,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x2c,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x2c,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x2c,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x2c,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x2c,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x2c,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x2c,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x2c,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x2c,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_nlg_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x0a,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x0a,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x0a,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x0a,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x0a,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x0a,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x0a,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x0a,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x0a,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x0a,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x0a,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x0a,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_nlg_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x1a,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x1a,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x1a,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x1a,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x1a,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x1a,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x1a,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x1a,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x1a,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x1a,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x1a,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x1a,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x1a,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_nlg_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x2a,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x2a,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x2a,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x2a,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x2a,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x2a,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x2a,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x2a,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x2a,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x2a,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x2a,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x2a,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x2a,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x2a,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x2a,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_nlt_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x0e,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x0e,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x0e,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x0e,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x0e,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x0e,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x0e,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x0e,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x0e,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x0e,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x0e,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x0e,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_nlt_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x1e,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x1e,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x1e,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x1e,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x1e,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x1e,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x1e,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x1e,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x1e,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x1e,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x1e,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x1e,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x1e,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_nlt_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x2e,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x2e,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x2e,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x2e,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x2e,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x2e,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x2e,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x2e,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x2e,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x2e,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x2e,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x2e,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x2e,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x2e,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x2e,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_o_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x07,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x07,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x07,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x07,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x07,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x07,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x07,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x07,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x07,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x07,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x07,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x07,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x07,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x07,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x07,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x07,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x07,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x07,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x07,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_o_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x17,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x17,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x17,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x17,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x17,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x17,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x17,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x17,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x17,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x17,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x17,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x17,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x17,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x17,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x17,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x17,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x17,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x17,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x17,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x17,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x17,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_o_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x27,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x27,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x27,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x27,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x27,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x27,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x27,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x27,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x27,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x27,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x27,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x27,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x27,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x27,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x27,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]

v_cmp_u_f16_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x08,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x08,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x08,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x08,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, vcc_hi, 0xfe0b
// W32: encoding: [0x05,0x00,0x08,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x08,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x08,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x08,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x08,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x08,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x08,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x08,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x08,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x08,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], vcc_hi, 0xfe0b
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x6b,0xfe,0x01,0x00,0x0b,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x08,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x08,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x08,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x08,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 null, -|0xfe0b|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x08,0xd4,0xff,0xd6,0x00,0x60,0x0b,0xfe,0x00,0x00]

v_cmp_u_f32_e64 s5, v1, v2
// W32: encoding: [0x05,0x00,0x18,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, v255, v255
// W32: encoding: [0x05,0x00,0x18,0xd4,0xff,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, s1, s2
// W32: encoding: [0x05,0x00,0x18,0xd4,0x01,0x04,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, s105, s105
// W32: encoding: [0x05,0x00,0x18,0xd4,0x69,0xd2,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, vcc_lo, ttmp15
// W32: encoding: [0x05,0x00,0x18,0xd4,0x6a,0xf6,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, vcc_hi, 0xaf123456
// W32: encoding: [0x05,0x00,0x18,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, ttmp15, src_scc
// W32: encoding: [0x05,0x00,0x18,0xd4,0x7b,0xfa,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, m0, 0.5
// W32: encoding: [0x05,0x00,0x18,0xd4,0x7d,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, exec_lo, -1
// W32: encoding: [0x05,0x00,0x18,0xd4,0x7e,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s5, |exec_hi|, null
// W32: encoding: [0x05,0x01,0x18,0xd4,0x7f,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s105, null, exec_lo
// W32: encoding: [0x69,0x00,0x18,0xd4,0x7c,0xfc,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 vcc_lo, -1, exec_hi
// W32: encoding: [0x6a,0x00,0x18,0xd4,0xc1,0xfe,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 vcc_hi, 0.5, -m0
// W32: encoding: [0x6b,0x00,0x18,0xd4,0xf0,0xfa,0x00,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 ttmp15, -src_scc, |vcc_lo|
// W32: encoding: [0x7b,0x02,0x18,0xd4,0xfd,0xd4,0x00,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v255, v255
// W64: encoding: [0x0a,0x00,0x18,0xd4,0xff,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], s1, s2
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x01,0x04,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], s105, s105
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x69,0xd2,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], vcc_lo, ttmp15
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x6a,0xf6,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], vcc_hi, 0xaf123456
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x6b,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], ttmp15, src_scc
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x7b,0xfa,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], m0, 0.5
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x7d,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], exec_lo, -1
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x7e,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], |exec_hi|, null
// W64: encoding: [0x0a,0x01,0x18,0xd4,0x7f,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], null, exec_lo
// W64: encoding: [0x0a,0x00,0x18,0xd4,0x7c,0xfc,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[104:105], -1, exec_hi
// W64: encoding: [0x68,0x00,0x18,0xd4,0xc1,0xfe,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 vcc, 0.5, -m0
// W64: encoding: [0x6a,0x00,0x18,0xd4,0xf0,0xfa,0x00,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 ttmp[14:15], -src_scc, |vcc_lo|
// W64: encoding: [0x7a,0x02,0x18,0xd4,0xfd,0xd4,0x00,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 null, -|0xaf123456|, -|vcc_hi| clamp
// GFX12: encoding: [0x7c,0x83,0x18,0xd4,0xff,0xd6,0x00,0x60,0x56,0x34,0x12,0xaf]

v_cmp_u_f64_e64 s5, v[1:2], v[2:3]
// W32: encoding: [0x05,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s5, v[254:255], v[254:255]
// W32: encoding: [0x05,0x00,0x28,0xd4,0xfe,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s5, s[2:3], s[4:5]
// W32: encoding: [0x05,0x00,0x28,0xd4,0x02,0x08,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s5, s[104:105], s[104:105]
// W32: encoding: [0x05,0x00,0x28,0xd4,0x68,0xd0,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s5, vcc, ttmp[14:15]
// W32: encoding: [0x05,0x00,0x28,0xd4,0x6a,0xf4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s5, ttmp[14:15], 0xaf123456
// W32: encoding: [0x05,0x00,0x28,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s5, -|exec|, src_scc
// W32: encoding: [0x05,0x01,0x28,0xd4,0x7e,0xfa,0x01,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s105, null, 0.5
// W32: encoding: [0x69,0x00,0x28,0xd4,0x7c,0xe0,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 vcc_lo, -1, -1
// W32: encoding: [0x6a,0x00,0x28,0xd4,0xc1,0x82,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 vcc_hi, 0.5, null
// W32: encoding: [0x6b,0x00,0x28,0xd4,0xf0,0xf8,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 ttmp15, -|src_scc|, -|exec|
// W32: encoding: [0x7b,0x03,0x28,0xd4,0xfd,0xfc,0x00,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[254:255], v[254:255]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0xfe,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], s[2:3], s[4:5]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x02,0x08,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], s[104:105], s[104:105]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x68,0xd0,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], vcc, ttmp[14:15]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x6a,0xf4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], ttmp[14:15], 0xaf123456
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x7a,0xfe,0x01,0x00,0x56,0x34,0x12,0xaf]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], -|exec|, src_scc
// W64: encoding: [0x0a,0x01,0x28,0xd4,0x7e,0xfa,0x01,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], null, 0.5
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x7c,0xe0,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[104:105], -1, -1
// W64: encoding: [0x68,0x00,0x28,0xd4,0xc1,0x82,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 vcc, 0.5, null
// W64: encoding: [0x6a,0x00,0x28,0xd4,0xf0,0xf8,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 ttmp[14:15], -|src_scc|, -|exec|
// W64: encoding: [0x7a,0x03,0x28,0xd4,0xfd,0xfc,0x00,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 null, 0xaf123456, -|vcc| clamp
// GFX12: encoding: [0x7c,0x82,0x28,0xd4,0xff,0xd4,0x00,0x40,0x56,0x34,0x12,0xaf]
