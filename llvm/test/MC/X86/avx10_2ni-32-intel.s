// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// VMPSADBW

// CHECK: vmpsadbw xmm2, xmm3, xmm4, 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0xd4,0x7b]
          vmpsadbw xmm2, xmm3, xmm4, 123

// CHECK: vmpsadbw xmm2 {k7}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x0f,0x42,0xd4,0x7b]
          vmpsadbw xmm2 {k7}, xmm3, xmm4, 123

// CHECK: vmpsadbw xmm2 {k7} {z}, xmm3, xmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x8f,0x42,0xd4,0x7b]
          vmpsadbw xmm2 {k7} {z}, xmm3, xmm4, 123

// CHECK: vmpsadbw ymm2, ymm3, ymm4, 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0xd4,0x7b]
          vmpsadbw ymm2, ymm3, ymm4, 123

// CHECK: vmpsadbw ymm2 {k7}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x2f,0x42,0xd4,0x7b]
          vmpsadbw ymm2 {k7}, ymm3, ymm4, 123

// CHECK: vmpsadbw ymm2 {k7} {z}, ymm3, ymm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0xaf,0x42,0xd4,0x7b]
          vmpsadbw ymm2 {k7} {z}, ymm3, ymm4, 123

// CHECK: vmpsadbw zmm2, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0xd4,0x7b]
          vmpsadbw zmm2, zmm3, zmm4, 123

// CHECK: vmpsadbw zmm2 {k7}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0x4f,0x42,0xd4,0x7b]
          vmpsadbw zmm2 {k7}, zmm3, zmm4, 123

// CHECK: vmpsadbw zmm2 {k7} {z}, zmm3, zmm4, 123
// CHECK: encoding: [0x62,0xf3,0x66,0xcf,0x42,0xd4,0x7b]
          vmpsadbw zmm2 {k7} {z}, zmm3, zmm4, 123

// CHECK: vmpsadbw xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw xmm2, xmm3, xmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vmpsadbw xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x0f,0x42,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw xmm2 {k7}, xmm3, xmmword ptr [edi + 4*eax + 291], 123

// CHECK: vmpsadbw xmm2, xmm3, xmmword ptr [eax], 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0x10,0x7b]
          vmpsadbw xmm2, xmm3, xmmword ptr [eax], 123

// CHECK: vmpsadbw xmm2, xmm3, xmmword ptr [2*ebp - 512], 123
// CHECK: encoding: [0xc4,0xe3,0x61,0x42,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vmpsadbw xmm2, xmm3, xmmword ptr [2*ebp - 512], 123

// CHECK: vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x8f,0x42,0x51,0x7f,0x7b]
          vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [ecx + 2032], 123

// CHECK: vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [edx - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x8f,0x42,0x52,0x80,0x7b]
          vmpsadbw xmm2 {k7} {z}, xmm3, xmmword ptr [edx - 2048], 123

// CHECK: vmpsadbw ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw ymm2, ymm3, ymmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vmpsadbw ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x2f,0x42,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw ymm2 {k7}, ymm3, ymmword ptr [edi + 4*eax + 291], 123

// CHECK: vmpsadbw ymm2, ymm3, ymmword ptr [eax], 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0x10,0x7b]
          vmpsadbw ymm2, ymm3, ymmword ptr [eax], 123

// CHECK: vmpsadbw ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123
// CHECK: encoding: [0xc4,0xe3,0x65,0x42,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vmpsadbw ymm2, ymm3, ymmword ptr [2*ebp - 1024], 123

// CHECK: vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xaf,0x42,0x51,0x7f,0x7b]
          vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [ecx + 4064], 123

// CHECK: vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [edx - 4096], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xaf,0x42,0x52,0x80,0x7b]
          vmpsadbw ymm2 {k7} {z}, ymm3, ymmword ptr [edx - 4096], 123

// CHECK: vmpsadbw zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vmpsadbw zmm2, zmm3, zmmword ptr [esp + 8*esi + 268435456], 123

// CHECK: vmpsadbw zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x4f,0x42,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vmpsadbw zmm2 {k7}, zmm3, zmmword ptr [edi + 4*eax + 291], 123

// CHECK: vmpsadbw zmm2, zmm3, zmmword ptr [eax], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0x10,0x7b]
          vmpsadbw zmm2, zmm3, zmmword ptr [eax], 123

// CHECK: vmpsadbw zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123
// CHECK: encoding: [0x62,0xf3,0x66,0x48,0x42,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vmpsadbw zmm2, zmm3, zmmword ptr [2*ebp - 2048], 123

// CHECK: vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xcf,0x42,0x51,0x7f,0x7b]
          vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [ecx + 8128], 123

// CHECK: vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [edx - 8192], 123
// CHECK: encoding: [0x62,0xf3,0x66,0xcf,0x42,0x52,0x80,0x7b]
          vmpsadbw zmm2 {k7} {z}, zmm3, zmmword ptr [edx - 8192], 123

// YMM Rounding

// CHECK: vaddpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x58,0xd4]
          vaddpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vaddpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x58,0xd4]
          vaddpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vaddpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x58,0xd4]
          vaddpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vaddph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x58,0xd4]
          vaddph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vaddph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x58,0xd4]
          vaddph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vaddph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x58,0xd4]
          vaddph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vaddps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x58,0xd4]
          vaddps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vaddps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x58,0xd4]
          vaddps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vaddps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x58,0xd4]
          vaddps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vcmppd k5, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0xc2,0xec,0x7b]
          vcmppd k5, ymm3, ymm4, {sae}, 123

// CHECK: vcmppd k5 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0xe1,0x1f,0xc2,0xec,0x7b]
          vcmppd k5 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vcmpph k5, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x60,0x18,0xc2,0xec,0x7b]
          vcmpph k5, ymm3, ymm4, {sae}, 123

// CHECK: vcmpph k5 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x60,0x1f,0xc2,0xec,0x7b]
          vcmpph k5 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vcmpps k5, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0xc2,0xec,0x7b]
          vcmpps k5, ymm3, ymm4, {sae}, 123

// CHECK: vcmpps k5 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf1,0x60,0x1f,0xc2,0xec,0x7b]
          vcmpps k5 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vcvtdq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x5b,0xd3]
          vcvtdq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtdq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x5b,0xd3]
          vcvtdq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtdq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x5b,0xd3]
          vcvtdq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtdq2ps ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x5b,0xd3]
          vcvtdq2ps ymm2, ymm3, {rn-sae}

// CHECK: vcvtdq2ps ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x3f,0x5b,0xd3]
          vcvtdq2ps ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtdq2ps ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0xff,0x5b,0xd3]
          vcvtdq2ps ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2dq xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x18,0xe6,0xd3]
          vcvtpd2dq xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2dq xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x3f,0xe6,0xd3]
          vcvtpd2dq xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2dq xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0xff,0xe6,0xd3]
          vcvtpd2dq xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x18,0x5a,0xd3]
          vcvtpd2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x3f,0x5a,0xd3]
          vcvtpd2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0xff,0x5a,0xd3]
          vcvtpd2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2ps xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x5a,0xd3]
          vcvtpd2ps xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2ps xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x5a,0xd3]
          vcvtpd2ps xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2ps xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x5a,0xd3]
          vcvtpd2ps xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2qq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x7b,0xd3]
          vcvtpd2qq ymm2, ymm3, {rn-sae}

// CHECK: vcvtpd2qq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x7b,0xd3]
          vcvtpd2qq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2qq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x7b,0xd3]
          vcvtpd2qq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2udq xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x18,0x79,0xd3]
          vcvtpd2udq xmm2, ymm3, {rn-sae}

// CHECK: vcvtpd2udq xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x3f,0x79,0xd3]
          vcvtpd2udq xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2udq xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0xff,0x79,0xd3]
          vcvtpd2udq xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtpd2uqq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x79,0xd3]
          vcvtpd2uqq ymm2, ymm3, {rn-sae}

// CHECK: vcvtpd2uqq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x79,0xd3]
          vcvtpd2uqq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtpd2uqq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x79,0xd3]
          vcvtpd2uqq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtph2dq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x5b,0xd3]
          vcvtph2dq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2dq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x5b,0xd3]
          vcvtph2dq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2dq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x5b,0xd3]
          vcvtph2dq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2pd ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x5a,0xd3]
          vcvtph2pd ymm2, xmm3, {sae}

// CHECK: vcvtph2pd ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x1f,0x5a,0xd3]
          vcvtph2pd ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtph2pd ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x5a,0xd3]
          vcvtph2pd ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtph2ps ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x18,0x13,0xd3]
          vcvtph2ps ymm2, xmm3, {sae}

// CHECK: vcvtph2ps ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x1f,0x13,0xd3]
          vcvtph2ps ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtph2ps ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x9f,0x13,0xd3]
          vcvtph2ps ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtph2psx ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x18,0x13,0xd3]
          vcvtph2psx ymm2, xmm3, {sae}

// CHECK: vcvtph2psx ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x1f,0x13,0xd3]
          vcvtph2psx ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtph2psx ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x9f,0x13,0xd3]
          vcvtph2psx ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtph2qq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7b,0xd3]
          vcvtph2qq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2qq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x7b,0xd3]
          vcvtph2qq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2qq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x7b,0xd3]
          vcvtph2qq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2udq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x79,0xd3]
          vcvtph2udq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2udq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x79,0xd3]
          vcvtph2udq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2udq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x79,0xd3]
          vcvtph2udq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2uqq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x79,0xd3]
          vcvtph2uqq ymm2, xmm3, {rn-sae}

// CHECK: vcvtph2uqq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x79,0xd3]
          vcvtph2uqq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtph2uqq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x79,0xd3]
          vcvtph2uqq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtph2uw ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x7d,0xd3]
          vcvtph2uw ymm2, ymm3, {rn-sae}

// CHECK: vcvtph2uw ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x7d,0xd3]
          vcvtph2uw ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtph2uw ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x7d,0xd3]
          vcvtph2uw ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtph2w ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7d,0xd3]
          vcvtph2w ymm2, ymm3, {rn-sae}

// CHECK: vcvtph2w ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x7d,0xd3]
          vcvtph2w ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtph2w ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x7d,0xd3]
          vcvtph2w ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2dq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x5b,0xd3]
          vcvtps2dq ymm2, ymm3, {rn-sae}

// CHECK: vcvtps2dq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x3f,0x5b,0xd3]
          vcvtps2dq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtps2dq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0xff,0x5b,0xd3]
          vcvtps2dq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2pd ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x5a,0xd3]
          vcvtps2pd ymm2, xmm3, {sae}

// CHECK: vcvtps2pd ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x1f,0x5a,0xd3]
          vcvtps2pd ymm2 {k7}, xmm3, {sae}

// CHECK: vcvtps2pd ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x9f,0x5a,0xd3]
          vcvtps2pd ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtps2ph xmm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x1d,0xda,0x7b]
          vcvtps2ph xmm2, ymm3, {sae}, 123

// CHECK: vcvtps2ph xmm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x1d,0xda,0x7b]
          vcvtps2ph xmm2 {k7}, ymm3, {sae}, 123

// CHECK: vcvtps2ph xmm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x1d,0xda,0x7b]
          vcvtps2ph xmm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vcvtps2phx xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x1d,0xd3]
          vcvtps2phx xmm2, ymm3, {rn-sae}

// CHECK: vcvtps2phx xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x3f,0x1d,0xd3]
          vcvtps2phx xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtps2phx xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x1d,0xd3]
          vcvtps2phx xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2qq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x7b,0xd3]
          vcvtps2qq ymm2, xmm3, {rn-sae}

// CHECK: vcvtps2qq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x3f,0x7b,0xd3]
          vcvtps2qq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtps2qq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0xff,0x7b,0xd3]
          vcvtps2qq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtps2udq ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x79,0xd3]
          vcvtps2udq ymm2, ymm3, {rn-sae}

// CHECK: vcvtps2udq ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x3f,0x79,0xd3]
          vcvtps2udq ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtps2udq ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0xff,0x79,0xd3]
          vcvtps2udq ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtps2uqq ymm2, xmm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x79,0xd3]
          vcvtps2uqq ymm2, xmm3, {rn-sae}

// CHECK: vcvtps2uqq ymm2 {k7}, xmm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x3f,0x79,0xd3]
          vcvtps2uqq ymm2 {k7}, xmm3, {rd-sae}

// CHECK: vcvtps2uqq ymm2 {k7} {z}, xmm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x79,0xff,0x79,0xd3]
          vcvtps2uqq ymm2 {k7} {z}, xmm3, {rz-sae}

// CHECK: vcvtqq2pd ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x18,0xe6,0xd3]
          vcvtqq2pd ymm2, ymm3, {rn-sae}

// CHECK: vcvtqq2pd ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x3f,0xe6,0xd3]
          vcvtqq2pd ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0xff,0xe6,0xd3]
          vcvtqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtqq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x18,0x5b,0xd3]
          vcvtqq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtqq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x3f,0x5b,0xd3]
          vcvtqq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0xff,0x5b,0xd3]
          vcvtqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtqq2ps xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x18,0x5b,0xd3]
          vcvtqq2ps xmm2, ymm3, {rn-sae}

// CHECK: vcvtqq2ps xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x3f,0x5b,0xd3]
          vcvtqq2ps xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0xff,0x5b,0xd3]
          vcvtqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvttpd2dq xmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0xe6,0xd3]
          vcvttpd2dq xmm2, ymm3, {sae}

// CHECK: vcvttpd2dq xmm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x1f,0xe6,0xd3]
          vcvttpd2dq xmm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2dq xmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x9f,0xe6,0xd3]
          vcvttpd2dq xmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2qq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x7a,0xd3]
          vcvttpd2qq ymm2, ymm3, {sae}

// CHECK: vcvttpd2qq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x1f,0x7a,0xd3]
          vcvttpd2qq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2qq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x9f,0x7a,0xd3]
          vcvttpd2qq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2udq xmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x18,0x78,0xd3]
          vcvttpd2udq xmm2, ymm3, {sae}

// CHECK: vcvttpd2udq xmm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x1f,0x78,0xd3]
          vcvttpd2udq xmm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2udq xmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf8,0x9f,0x78,0xd3]
          vcvttpd2udq xmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2uqq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x78,0xd3]
          vcvttpd2uqq ymm2, ymm3, {sae}

// CHECK: vcvttpd2uqq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x1f,0x78,0xd3]
          vcvttpd2uqq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttpd2uqq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x9f,0x78,0xd3]
          vcvttpd2uqq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttph2dq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x18,0x5b,0xd3]
          vcvttph2dq ymm2, xmm3, {sae}

// CHECK: vcvttph2dq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x1f,0x5b,0xd3]
          vcvttph2dq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2dq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x9f,0x5b,0xd3]
          vcvttph2dq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2qq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7a,0xd3]
          vcvttph2qq ymm2, xmm3, {sae}

// CHECK: vcvttph2qq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x1f,0x7a,0xd3]
          vcvttph2qq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2qq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x7a,0xd3]
          vcvttph2qq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2udq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x78,0xd3]
          vcvttph2udq ymm2, xmm3, {sae}

// CHECK: vcvttph2udq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x1f,0x78,0xd3]
          vcvttph2udq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2udq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x78,0xd3]
          vcvttph2udq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2uqq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x78,0xd3]
          vcvttph2uqq ymm2, xmm3, {sae}

// CHECK: vcvttph2uqq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x1f,0x78,0xd3]
          vcvttph2uqq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttph2uqq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x78,0xd3]
          vcvttph2uqq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttph2uw ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x7c,0xd3]
          vcvttph2uw ymm2, ymm3, {sae}

// CHECK: vcvttph2uw ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x1f,0x7c,0xd3]
          vcvttph2uw ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttph2uw ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x7c,0xd3]
          vcvttph2uw ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttph2w ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x7c,0xd3]
          vcvttph2w ymm2, ymm3, {sae}

// CHECK: vcvttph2w ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x1f,0x7c,0xd3]
          vcvttph2w ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttph2w ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x7c,0xd3]
          vcvttph2w ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2dq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x7a,0x18,0x5b,0xd3]
          vcvttps2dq ymm2, ymm3, {sae}

// CHECK: vcvttps2dq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x7a,0x1f,0x5b,0xd3]
          vcvttps2dq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttps2dq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x7a,0x9f,0x5b,0xd3]
          vcvttps2dq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2qq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x7a,0xd3]
          vcvttps2qq ymm2, xmm3, {sae}

// CHECK: vcvttps2qq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x1f,0x7a,0xd3]
          vcvttps2qq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttps2qq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x9f,0x7a,0xd3]
          vcvttps2qq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttps2udq ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x78,0xd3]
          vcvttps2udq ymm2, ymm3, {sae}

// CHECK: vcvttps2udq ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x1f,0x78,0xd3]
          vcvttps2udq ymm2 {k7}, ymm3, {sae}

// CHECK: vcvttps2udq ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x9f,0x78,0xd3]
          vcvttps2udq ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2uqq ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x18,0x78,0xd3]
          vcvttps2uqq ymm2, xmm3, {sae}

// CHECK: vcvttps2uqq ymm2 {k7}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x1f,0x78,0xd3]
          vcvttps2uqq ymm2 {k7}, xmm3, {sae}

// CHECK: vcvttps2uqq ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf1,0x79,0x9f,0x78,0xd3]
          vcvttps2uqq ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvtudq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x18,0x7a,0xd3]
          vcvtudq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtudq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x3f,0x7a,0xd3]
          vcvtudq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtudq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0xff,0x7a,0xd3]
          vcvtudq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtudq2ps ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x7b,0x18,0x7a,0xd3]
          vcvtudq2ps ymm2, ymm3, {rn-sae}

// CHECK: vcvtudq2ps ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x7b,0x3f,0x7a,0xd3]
          vcvtudq2ps ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtudq2ps ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x7b,0xff,0x7a,0xd3]
          vcvtudq2ps ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuqq2pd ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x18,0x7a,0xd3]
          vcvtuqq2pd ymm2, ymm3, {rn-sae}

// CHECK: vcvtuqq2pd ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0x3f,0x7a,0xd3]
          vcvtuqq2pd ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfa,0xff,0x7a,0xd3]
          vcvtuqq2pd ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuqq2ph xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0xfb,0x18,0x7a,0xd3]
          vcvtuqq2ph xmm2, ymm3, {rn-sae}

// CHECK: vcvtuqq2ph xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0xfb,0x3f,0x7a,0xd3]
          vcvtuqq2ph xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0xfb,0xff,0x7a,0xd3]
          vcvtuqq2ph xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuqq2ps xmm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x18,0x7a,0xd3]
          vcvtuqq2ps xmm2, ymm3, {rn-sae}

// CHECK: vcvtuqq2ps xmm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0x3f,0x7a,0xd3]
          vcvtuqq2ps xmm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xfb,0xff,0x7a,0xd3]
          vcvtuqq2ps xmm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtuw2ph ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x18,0x7d,0xd3]
          vcvtuw2ph ymm2, ymm3, {rn-sae}

// CHECK: vcvtuw2ph ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0x3f,0x7d,0xd3]
          vcvtuw2ph ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtuw2ph ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7b,0xff,0x7d,0xd3]
          vcvtuw2ph ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vcvtw2ph ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x18,0x7d,0xd3]
          vcvtw2ph ymm2, ymm3, {rn-sae}

// CHECK: vcvtw2ph ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0x3f,0x7d,0xd3]
          vcvtw2ph ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vcvtw2ph ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x7a,0xff,0x7d,0xd3]
          vcvtw2ph ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vdivpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5e,0xd4]
          vdivpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vdivpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x5e,0xd4]
          vdivpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vdivpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x5e,0xd4]
          vdivpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vdivph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5e,0xd4]
          vdivph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vdivph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x5e,0xd4]
          vdivph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vdivph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x5e,0xd4]
          vdivph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vdivps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5e,0xd4]
          vdivps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vdivps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x5e,0xd4]
          vdivps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vdivps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x5e,0xd4]
          vdivps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfcmaddcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x18,0x56,0xd4]
          vfcmaddcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfcmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x3f,0x56,0xd4]
          vfcmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfcmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0xff,0x56,0xd4]
          vfcmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfcmulcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x18,0xd6,0xd4]
          vfcmulcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfcmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0x3f,0xd6,0xd4]
          vfcmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfcmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x63,0xff,0xd6,0xd4]
          vfcmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfixupimmpd ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x18,0x54,0xd4,0x7b]
          vfixupimmpd ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmpd ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x1f,0x54,0xd4,0x7b]
          vfixupimmpd ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmpd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x9f,0x54,0xd4,0x7b]
          vfixupimmpd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmps ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x18,0x54,0xd4,0x7b]
          vfixupimmps ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmps ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x1f,0x54,0xd4,0x7b]
          vfixupimmps ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vfixupimmps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x9f,0x54,0xd4,0x7b]
          vfixupimmps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vfmadd132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x98,0xd4]
          vfmadd132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x98,0xd4]
          vfmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x98,0xd4]
          vfmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x98,0xd4]
          vfmadd132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x98,0xd4]
          vfmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x98,0xd4]
          vfmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x98,0xd4]
          vfmadd132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x98,0xd4]
          vfmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x98,0xd4]
          vfmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xa8,0xd4]
          vfmadd213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xa8,0xd4]
          vfmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xa8,0xd4]
          vfmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xa8,0xd4]
          vfmadd213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xa8,0xd4]
          vfmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xa8,0xd4]
          vfmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xa8,0xd4]
          vfmadd213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xa8,0xd4]
          vfmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xa8,0xd4]
          vfmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xb8,0xd4]
          vfmadd231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xb8,0xd4]
          vfmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xb8,0xd4]
          vfmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xb8,0xd4]
          vfmadd231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xb8,0xd4]
          vfmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xb8,0xd4]
          vfmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmadd231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xb8,0xd4]
          vfmadd231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xb8,0xd4]
          vfmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xb8,0xd4]
          vfmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x18,0x56,0xd4]
          vfmaddcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x3f,0x56,0xd4]
          vfmaddcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0xff,0x56,0xd4]
          vfmaddcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x96,0xd4]
          vfmaddsub132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x96,0xd4]
          vfmaddsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x96,0xd4]
          vfmaddsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x96,0xd4]
          vfmaddsub132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x96,0xd4]
          vfmaddsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x96,0xd4]
          vfmaddsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x96,0xd4]
          vfmaddsub132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x96,0xd4]
          vfmaddsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x96,0xd4]
          vfmaddsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xa6,0xd4]
          vfmaddsub213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xa6,0xd4]
          vfmaddsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xa6,0xd4]
          vfmaddsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xa6,0xd4]
          vfmaddsub213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xa6,0xd4]
          vfmaddsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xa6,0xd4]
          vfmaddsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xa6,0xd4]
          vfmaddsub213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xa6,0xd4]
          vfmaddsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xa6,0xd4]
          vfmaddsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xb6,0xd4]
          vfmaddsub231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xb6,0xd4]
          vfmaddsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xb6,0xd4]
          vfmaddsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xb6,0xd4]
          vfmaddsub231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xb6,0xd4]
          vfmaddsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xb6,0xd4]
          vfmaddsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmaddsub231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xb6,0xd4]
          vfmaddsub231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmaddsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xb6,0xd4]
          vfmaddsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmaddsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xb6,0xd4]
          vfmaddsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x9a,0xd4]
          vfmsub132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x9a,0xd4]
          vfmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x9a,0xd4]
          vfmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x9a,0xd4]
          vfmsub132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x9a,0xd4]
          vfmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x9a,0xd4]
          vfmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x9a,0xd4]
          vfmsub132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x9a,0xd4]
          vfmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x9a,0xd4]
          vfmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xaa,0xd4]
          vfmsub213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xaa,0xd4]
          vfmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xaa,0xd4]
          vfmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xaa,0xd4]
          vfmsub213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xaa,0xd4]
          vfmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xaa,0xd4]
          vfmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xaa,0xd4]
          vfmsub213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xaa,0xd4]
          vfmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xaa,0xd4]
          vfmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xba,0xd4]
          vfmsub231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xba,0xd4]
          vfmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xba,0xd4]
          vfmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xba,0xd4]
          vfmsub231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xba,0xd4]
          vfmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xba,0xd4]
          vfmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsub231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xba,0xd4]
          vfmsub231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xba,0xd4]
          vfmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xba,0xd4]
          vfmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x97,0xd4]
          vfmsubadd132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x97,0xd4]
          vfmsubadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x97,0xd4]
          vfmsubadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x97,0xd4]
          vfmsubadd132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x97,0xd4]
          vfmsubadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x97,0xd4]
          vfmsubadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x97,0xd4]
          vfmsubadd132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x97,0xd4]
          vfmsubadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x97,0xd4]
          vfmsubadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xa7,0xd4]
          vfmsubadd213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xa7,0xd4]
          vfmsubadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xa7,0xd4]
          vfmsubadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xa7,0xd4]
          vfmsubadd213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xa7,0xd4]
          vfmsubadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xa7,0xd4]
          vfmsubadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xa7,0xd4]
          vfmsubadd213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xa7,0xd4]
          vfmsubadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xa7,0xd4]
          vfmsubadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xb7,0xd4]
          vfmsubadd231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xb7,0xd4]
          vfmsubadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xb7,0xd4]
          vfmsubadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xb7,0xd4]
          vfmsubadd231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xb7,0xd4]
          vfmsubadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xb7,0xd4]
          vfmsubadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmsubadd231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xb7,0xd4]
          vfmsubadd231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmsubadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xb7,0xd4]
          vfmsubadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmsubadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xb7,0xd4]
          vfmsubadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfmulcph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x18,0xd6,0xd4]
          vfmulcph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0x3f,0xd6,0xd4]
          vfmulcph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x62,0xff,0xd6,0xd4]
          vfmulcph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x9c,0xd4]
          vfnmadd132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x9c,0xd4]
          vfnmadd132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x9c,0xd4]
          vfnmadd132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x9c,0xd4]
          vfnmadd132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x9c,0xd4]
          vfnmadd132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x9c,0xd4]
          vfnmadd132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x9c,0xd4]
          vfnmadd132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x9c,0xd4]
          vfnmadd132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x9c,0xd4]
          vfnmadd132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xac,0xd4]
          vfnmadd213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xac,0xd4]
          vfnmadd213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xac,0xd4]
          vfnmadd213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xac,0xd4]
          vfnmadd213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xac,0xd4]
          vfnmadd213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xac,0xd4]
          vfnmadd213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xac,0xd4]
          vfnmadd213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xac,0xd4]
          vfnmadd213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xac,0xd4]
          vfnmadd213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xbc,0xd4]
          vfnmadd231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xbc,0xd4]
          vfnmadd231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xbc,0xd4]
          vfnmadd231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xbc,0xd4]
          vfnmadd231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xbc,0xd4]
          vfnmadd231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xbc,0xd4]
          vfnmadd231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmadd231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xbc,0xd4]
          vfnmadd231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xbc,0xd4]
          vfnmadd231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xbc,0xd4]
          vfnmadd231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub132pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x9e,0xd4]
          vfnmsub132pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x9e,0xd4]
          vfnmsub132pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x9e,0xd4]
          vfnmsub132pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub132ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x9e,0xd4]
          vfnmsub132ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x9e,0xd4]
          vfnmsub132ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x9e,0xd4]
          vfnmsub132ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub132ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x9e,0xd4]
          vfnmsub132ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x9e,0xd4]
          vfnmsub132ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x9e,0xd4]
          vfnmsub132ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub213pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xae,0xd4]
          vfnmsub213pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xae,0xd4]
          vfnmsub213pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xae,0xd4]
          vfnmsub213pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub213ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xae,0xd4]
          vfnmsub213ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xae,0xd4]
          vfnmsub213ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xae,0xd4]
          vfnmsub213ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub213ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xae,0xd4]
          vfnmsub213ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xae,0xd4]
          vfnmsub213ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xae,0xd4]
          vfnmsub213ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub231pd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0xbe,0xd4]
          vfnmsub231pd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0xbe,0xd4]
          vfnmsub231pd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0xbe,0xd4]
          vfnmsub231pd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub231ph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0xbe,0xd4]
          vfnmsub231ph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0xbe,0xd4]
          vfnmsub231ph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0xbe,0xd4]
          vfnmsub231ph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vfnmsub231ps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0xbe,0xd4]
          vfnmsub231ps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vfnmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0xbe,0xd4]
          vfnmsub231ps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vfnmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0xbe,0xd4]
          vfnmsub231ps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vgetexppd ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0xf9,0x18,0x42,0xd3]
          vgetexppd ymm2, ymm3, {sae}

// CHECK: vgetexppd ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0xf9,0x1f,0x42,0xd3]
          vgetexppd ymm2 {k7}, ymm3, {sae}

// CHECK: vgetexppd ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0xf9,0x9f,0x42,0xd3]
          vgetexppd ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vgetexpph ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x18,0x42,0xd3]
          vgetexpph ymm2, ymm3, {sae}

// CHECK: vgetexpph ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x1f,0x42,0xd3]
          vgetexpph ymm2 {k7}, ymm3, {sae}

// CHECK: vgetexpph ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf6,0x79,0x9f,0x42,0xd3]
          vgetexpph ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vgetexpps ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x18,0x42,0xd3]
          vgetexpps ymm2, ymm3, {sae}

// CHECK: vgetexpps ymm2 {k7}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x1f,0x42,0xd3]
          vgetexpps ymm2 {k7}, ymm3, {sae}

// CHECK: vgetexpps ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf2,0x79,0x9f,0x42,0xd3]
          vgetexpps ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vgetmantpd ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x18,0x26,0xd3,0x7b]
          vgetmantpd ymm2, ymm3, {sae}, 123

// CHECK: vgetmantpd ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x1f,0x26,0xd3,0x7b]
          vgetmantpd ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vgetmantpd ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x9f,0x26,0xd3,0x7b]
          vgetmantpd ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vgetmantph ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x18,0x26,0xd3,0x7b]
          vgetmantph ymm2, ymm3, {sae}, 123

// CHECK: vgetmantph ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x1f,0x26,0xd3,0x7b]
          vgetmantph ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vgetmantph ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x9f,0x26,0xd3,0x7b]
          vgetmantph ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vgetmantps ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x26,0xd3,0x7b]
          vgetmantps ymm2, ymm3, {sae}, 123

// CHECK: vgetmantps ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x26,0xd3,0x7b]
          vgetmantps ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vgetmantps ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x26,0xd3,0x7b]
          vgetmantps ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vmaxpd ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5f,0xd4]
          vmaxpd ymm2, ymm3, ymm4, {sae}

// CHECK: vmaxpd ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x1f,0x5f,0xd4]
          vmaxpd ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vmaxpd ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x9f,0x5f,0xd4]
          vmaxpd ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vmaxph ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5f,0xd4]
          vmaxph ymm2, ymm3, ymm4, {sae}

// CHECK: vmaxph ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x1f,0x5f,0xd4]
          vmaxph ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vmaxph ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x9f,0x5f,0xd4]
          vmaxph ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vmaxps ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5f,0xd4]
          vmaxps ymm2, ymm3, ymm4, {sae}

// CHECK: vmaxps ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x1f,0x5f,0xd4]
          vmaxps ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vmaxps ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x9f,0x5f,0xd4]
          vmaxps ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vminpd ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5d,0xd4]
          vminpd ymm2, ymm3, ymm4, {sae}

// CHECK: vminpd ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x1f,0x5d,0xd4]
          vminpd ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vminpd ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x9f,0x5d,0xd4]
          vminpd ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vminph ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5d,0xd4]
          vminph ymm2, ymm3, ymm4, {sae}

// CHECK: vminph ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x1f,0x5d,0xd4]
          vminph ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vminph ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x9f,0x5d,0xd4]
          vminph ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vminps ymm2, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5d,0xd4]
          vminps ymm2, ymm3, ymm4, {sae}

// CHECK: vminps ymm2 {k7}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x1f,0x5d,0xd4]
          vminps ymm2 {k7}, ymm3, ymm4, {sae}

// CHECK: vminps ymm2 {k7} {z}, ymm3, ymm4, {sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x9f,0x5d,0xd4]
          vminps ymm2 {k7} {z}, ymm3, ymm4, {sae}

// CHECK: vmulpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x59,0xd4]
          vmulpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vmulpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x59,0xd4]
          vmulpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vmulpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x59,0xd4]
          vmulpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vmulph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x59,0xd4]
          vmulph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vmulph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x59,0xd4]
          vmulph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vmulph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x59,0xd4]
          vmulph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vmulps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x59,0xd4]
          vmulps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vmulps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x59,0xd4]
          vmulps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vmulps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x59,0xd4]
          vmulps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vrangepd ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x18,0x50,0xd4,0x7b]
          vrangepd ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vrangepd ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x1f,0x50,0xd4,0x7b]
          vrangepd ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vrangepd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xe1,0x9f,0x50,0xd4,0x7b]
          vrangepd ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vrangeps ymm2, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x18,0x50,0xd4,0x7b]
          vrangeps ymm2, ymm3, ymm4, {sae}, 123

// CHECK: vrangeps ymm2 {k7}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x1f,0x50,0xd4,0x7b]
          vrangeps ymm2 {k7}, ymm3, ymm4, {sae}, 123

// CHECK: vrangeps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x61,0x9f,0x50,0xd4,0x7b]
          vrangeps ymm2 {k7} {z}, ymm3, ymm4, {sae}, 123

// CHECK: vreducepd ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x18,0x56,0xd3,0x7b]
          vreducepd ymm2, ymm3, {sae}, 123

// CHECK: vreducepd ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x1f,0x56,0xd3,0x7b]
          vreducepd ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vreducepd ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x9f,0x56,0xd3,0x7b]
          vreducepd ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vreduceph ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x18,0x56,0xd3,0x7b]
          vreduceph ymm2, ymm3, {sae}, 123

// CHECK: vreduceph ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x1f,0x56,0xd3,0x7b]
          vreduceph ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vreduceph ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x9f,0x56,0xd3,0x7b]
          vreduceph ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vreduceps ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x56,0xd3,0x7b]
          vreduceps ymm2, ymm3, {sae}, 123

// CHECK: vreduceps ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x56,0xd3,0x7b]
          vreduceps ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vreduceps ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x56,0xd3,0x7b]
          vreduceps ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vrndscalepd ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x18,0x09,0xd3,0x7b]
          vrndscalepd ymm2, ymm3, {sae}, 123

// CHECK: vrndscalepd ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x1f,0x09,0xd3,0x7b]
          vrndscalepd ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vrndscalepd ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0xf9,0x9f,0x09,0xd3,0x7b]
          vrndscalepd ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vrndscaleph ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x18,0x08,0xd3,0x7b]
          vrndscaleph ymm2, ymm3, {sae}, 123

// CHECK: vrndscaleph ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x1f,0x08,0xd3,0x7b]
          vrndscaleph ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vrndscaleph ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x78,0x9f,0x08,0xd3,0x7b]
          vrndscaleph ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vrndscaleps ymm2, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x18,0x08,0xd3,0x7b]
          vrndscaleps ymm2, ymm3, {sae}, 123

// CHECK: vrndscaleps ymm2 {k7}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x1f,0x08,0xd3,0x7b]
          vrndscaleps ymm2 {k7}, ymm3, {sae}, 123

// CHECK: vrndscaleps ymm2 {k7} {z}, ymm3, {sae}, 123
// CHECK: encoding: [0x62,0xf3,0x79,0x9f,0x08,0xd3,0x7b]
          vrndscaleps ymm2 {k7} {z}, ymm3, {sae}, 123

// CHECK: vscalefpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x18,0x2c,0xd4]
          vscalefpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vscalefpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0x3f,0x2c,0xd4]
          vscalefpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vscalefpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0xe1,0xff,0x2c,0xd4]
          vscalefpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vscalefph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x18,0x2c,0xd4]
          vscalefph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vscalefph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0x3f,0x2c,0xd4]
          vscalefph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vscalefph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf6,0x61,0xff,0x2c,0xd4]
          vscalefph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vscalefps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x2c,0xd4]
          vscalefps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vscalefps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0x3f,0x2c,0xd4]
          vscalefps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vscalefps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x2c,0xd4]
          vscalefps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vsqrtpd ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x18,0x51,0xd3]
          vsqrtpd ymm2, ymm3, {rn-sae}

// CHECK: vsqrtpd ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0x3f,0x51,0xd3]
          vsqrtpd ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vsqrtpd ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xf9,0xff,0x51,0xd3]
          vsqrtpd ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vsqrtph ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x51,0xd3]
          vsqrtph ymm2, ymm3, {rn-sae}

// CHECK: vsqrtph ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x3f,0x51,0xd3]
          vsqrtph ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vsqrtph ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x51,0xd3]
          vsqrtph ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vsqrtps ymm2, ymm3, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x18,0x51,0xd3]
          vsqrtps ymm2, ymm3, {rn-sae}

// CHECK: vsqrtps ymm2 {k7}, ymm3, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0x3f,0x51,0xd3]
          vsqrtps ymm2 {k7}, ymm3, {rd-sae}

// CHECK: vsqrtps ymm2 {k7} {z}, ymm3, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x78,0xff,0x51,0xd3]
          vsqrtps ymm2 {k7} {z}, ymm3, {rz-sae}

// CHECK: vsubpd ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x18,0x5c,0xd4]
          vsubpd ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vsubpd ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0x3f,0x5c,0xd4]
          vsubpd ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vsubpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0xe1,0xff,0x5c,0xd4]
          vsubpd ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vsubph ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x18,0x5c,0xd4]
          vsubph ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vsubph ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0x3f,0x5c,0xd4]
          vsubph ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vsubph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf5,0x60,0xff,0x5c,0xd4]
          vsubph ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}

// CHECK: vsubps ymm2, ymm3, ymm4, {rn-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x18,0x5c,0xd4]
          vsubps ymm2, ymm3, ymm4, {rn-sae}

// CHECK: vsubps ymm2 {k7}, ymm3, ymm4, {rd-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0x3f,0x5c,0xd4]
          vsubps ymm2 {k7}, ymm3, ymm4, {rd-sae}

// CHECK: vsubps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
// CHECK: encoding: [0x62,0xf1,0x60,0xff,0x5c,0xd4]
          vsubps ymm2 {k7} {z}, ymm3, ymm4, {rz-sae}
