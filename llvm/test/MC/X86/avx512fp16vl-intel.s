// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vaddph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x05,0x14,0x20,0x58,0xf4]
          vaddph ymm30, ymm29, ymm28

// CHECK: vaddph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x05,0x14,0x00,0x58,0xf4]
          vaddph xmm30, xmm29, xmm28

// CHECK: vaddph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x27,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x14,0x30,0x58,0x31]
          vaddph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vaddph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x14,0x20,0x58,0x71,0x7f]
          vaddph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vaddph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x14,0xb7,0x58,0x72,0x80]
          vaddph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vaddph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x07,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vaddph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x14,0x10,0x58,0x31]
          vaddph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vaddph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x14,0x00,0x58,0x71,0x7f]
          vaddph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vaddph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x14,0x97,0x58,0x72,0x80]
          vaddph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vcmpeqph k5, ymm29, ymm28
// CHECK: encoding: [0x62,0x93,0x14,0x20,0xc2,0xec,0x00]
          vcmpph k5, ymm29, ymm28, 0

// CHECK: vcmpltph k5, xmm29, xmm28
// CHECK: encoding: [0x62,0x93,0x14,0x00,0xc2,0xec,0x01]
          vcmpph k5, xmm29, xmm28, 1

// CHECK: vcmpleph k5 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb3,0x14,0x07,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x02]
          vcmpph k5 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456], 2

// CHECK: vcmpunordph k5, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0xd3,0x14,0x10,0xc2,0x29,0x03]
          vcmpph k5, xmm29, word ptr [r9]{1to8}, 3

// CHECK: vcmpneqph k5, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xf3,0x14,0x00,0xc2,0x69,0x7f,0x04]
          vcmpph k5, xmm29, xmmword ptr [rcx + 2032], 4

// CHECK: vcmpnltph k5 {k7}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0xf3,0x14,0x17,0xc2,0x6a,0x80,0x05]
          vcmpph k5 {k7}, xmm29, word ptr [rdx - 256]{1to8}, 5

// CHECK: vcmpnleph k5 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb3,0x14,0x27,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x06]
          vcmpph k5 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456], 6

// CHECK: vcmpordph k5, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0xd3,0x14,0x30,0xc2,0x29,0x07]
          vcmpph k5, ymm29, word ptr [r9]{1to16}, 7

// CHECK: vcmpeq_uqph k5, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xf3,0x14,0x20,0xc2,0x69,0x7f,0x08]
          vcmpph k5, ymm29, ymmword ptr [rcx + 4064], 8

// CHECK: vcmpngeph k5 {k7}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0xf3,0x14,0x37,0xc2,0x6a,0x80,0x09]
          vcmpph k5 {k7}, ymm29, word ptr [rdx - 256]{1to16}, 9

// CHECK: vdivph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x05,0x14,0x20,0x5e,0xf4]
          vdivph ymm30, ymm29, ymm28

// CHECK: vdivph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x05,0x14,0x00,0x5e,0xf4]
          vdivph xmm30, xmm29, xmm28

// CHECK: vdivph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x27,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x14,0x30,0x5e,0x31]
          vdivph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vdivph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x14,0x20,0x5e,0x71,0x7f]
          vdivph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vdivph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x14,0xb7,0x5e,0x72,0x80]
          vdivph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vdivph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x07,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vdivph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x14,0x10,0x5e,0x31]
          vdivph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vdivph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x14,0x00,0x5e,0x71,0x7f]
          vdivph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vdivph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x14,0x97,0x5e,0x72,0x80]
          vdivph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vmaxph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x05,0x14,0x20,0x5f,0xf4]
          vmaxph ymm30, ymm29, ymm28

// CHECK: vmaxph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x05,0x14,0x00,0x5f,0xf4]
          vmaxph xmm30, xmm29, xmm28

// CHECK: vmaxph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x27,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x14,0x30,0x5f,0x31]
          vmaxph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vmaxph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x14,0x20,0x5f,0x71,0x7f]
          vmaxph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vmaxph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x14,0xb7,0x5f,0x72,0x80]
          vmaxph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vmaxph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x07,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmaxph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x14,0x10,0x5f,0x31]
          vmaxph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vmaxph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x14,0x00,0x5f,0x71,0x7f]
          vmaxph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vmaxph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x14,0x97,0x5f,0x72,0x80]
          vmaxph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vminph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x05,0x14,0x20,0x5d,0xf4]
          vminph ymm30, ymm29, ymm28

// CHECK: vminph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x05,0x14,0x00,0x5d,0xf4]
          vminph xmm30, xmm29, xmm28

// CHECK: vminph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x27,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x14,0x30,0x5d,0x31]
          vminph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vminph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x14,0x20,0x5d,0x71,0x7f]
          vminph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vminph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x14,0xb7,0x5d,0x72,0x80]
          vminph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vminph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x07,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vminph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x14,0x10,0x5d,0x31]
          vminph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vminph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x14,0x00,0x5d,0x71,0x7f]
          vminph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vminph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x14,0x97,0x5d,0x72,0x80]
          vminph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vmulph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x05,0x14,0x20,0x59,0xf4]
          vmulph ymm30, ymm29, ymm28

// CHECK: vmulph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x05,0x14,0x00,0x59,0xf4]
          vmulph xmm30, xmm29, xmm28

// CHECK: vmulph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x27,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x14,0x30,0x59,0x31]
          vmulph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vmulph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x14,0x20,0x59,0x71,0x7f]
          vmulph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vmulph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x14,0xb7,0x59,0x72,0x80]
          vmulph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vmulph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x07,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vmulph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x14,0x10,0x59,0x31]
          vmulph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vmulph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x14,0x00,0x59,0x71,0x7f]
          vmulph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vmulph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x14,0x97,0x59,0x72,0x80]
          vmulph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vsubph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x05,0x14,0x20,0x5c,0xf4]
          vsubph ymm30, ymm29, ymm28

// CHECK: vsubph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x05,0x14,0x00,0x5c,0xf4]
          vsubph xmm30, xmm29, xmm28

// CHECK: vsubph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x27,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x14,0x30,0x5c,0x31]
          vsubph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vsubph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x14,0x20,0x5c,0x71,0x7f]
          vsubph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vsubph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x14,0xb7,0x5c,0x72,0x80]
          vsubph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vsubph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x14,0x07,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsubph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x14,0x10,0x5c,0x31]
          vsubph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vsubph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x14,0x00,0x5c,0x71,0x7f]
          vsubph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vsubph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x14,0x97,0x5c,0x72,0x80]
          vsubph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vcvtdq2ph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x5b,0xf5]
          vcvtdq2ph xmm30, xmm29

// CHECK: vcvtdq2ph xmm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7c,0x28,0x5b,0xf5]
          vcvtdq2ph xmm30, ymm29

// CHECK: vcvtdq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x0f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtdq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtdq2ph xmm30, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7c,0x18,0x5b,0x31]
          vcvtdq2ph xmm30, dword ptr [r9]{1to4}

// CHECK: vcvtdq2ph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x5b,0x71,0x7f]
          vcvtdq2ph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtdq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x65,0x7c,0x9f,0x5b,0x72,0x80]
          vcvtdq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtdq2ph xmm30, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7c,0x38,0x5b,0x31]
          vcvtdq2ph xmm30, dword ptr [r9]{1to8}

// CHECK: vcvtdq2ph xmm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7c,0x28,0x5b,0x71,0x7f]
          vcvtdq2ph xmm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtdq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x65,0x7c,0xbf,0x5b,0x72,0x80]
          vcvtdq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvtpd2ph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0xfd,0x08,0x5a,0xf5]
          vcvtpd2ph xmm30, xmm29

// CHECK: vcvtpd2ph xmm30, ymm29
// CHECK: encoding: [0x62,0x05,0xfd,0x28,0x5a,0xf5]
          vcvtpd2ph xmm30, ymm29

// CHECK: vcvtpd2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0xfd,0x0f,0x5a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtpd2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtpd2ph xmm30, qword ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0xfd,0x18,0x5a,0x31]
          vcvtpd2ph xmm30, qword ptr [r9]{1to2}

// CHECK: vcvtpd2ph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0xfd,0x08,0x5a,0x71,0x7f]
          vcvtpd2ph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtpd2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0x65,0xfd,0x9f,0x5a,0x72,0x80]
          vcvtpd2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to2}

// CHECK: vcvtpd2ph xmm30, qword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0xfd,0x38,0x5a,0x31]
          vcvtpd2ph xmm30, qword ptr [r9]{1to4}

// CHECK: vcvtpd2ph xmm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0xfd,0x28,0x5a,0x71,0x7f]
          vcvtpd2ph xmm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtpd2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0x65,0xfd,0xbf,0x5a,0x72,0x80]
          vcvtpd2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to4}

// CHECK: vcvtph2dq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x5b,0xf5]
          vcvtph2dq xmm30, xmm29

// CHECK: vcvtph2dq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x5b,0xf5]
          vcvtph2dq ymm30, xmm29

// CHECK: vcvtph2dq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2dq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2dq xmm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x5b,0x31]
          vcvtph2dq xmm30, word ptr [r9]{1to4}

// CHECK: vcvtph2dq xmm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x5b,0x71,0x7f]
          vcvtph2dq xmm30, qword ptr [rcx + 1016]

// CHECK: vcvtph2dq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x5b,0x72,0x80]
          vcvtph2dq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvtph2dq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x2f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2dq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2dq ymm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x5b,0x31]
          vcvtph2dq ymm30, word ptr [r9]{1to8}

// CHECK: vcvtph2dq ymm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x5b,0x71,0x7f]
          vcvtph2dq ymm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtph2dq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x5b,0x72,0x80]
          vcvtph2dq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtph2pd xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x5a,0xf5]
          vcvtph2pd xmm30, xmm29

// CHECK: vcvtph2pd ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x28,0x5a,0xf5]
          vcvtph2pd ymm30, xmm29

// CHECK: vcvtph2pd xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x0f,0x5a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2pd xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2pd xmm30, word ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0x7c,0x18,0x5a,0x31]
          vcvtph2pd xmm30, word ptr [r9]{1to2}

// CHECK: vcvtph2pd xmm30, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x5a,0x71,0x7f]
          vcvtph2pd xmm30, dword ptr [rcx + 508]

// CHECK: vcvtph2pd xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}
// CHECK: encoding: [0x62,0x65,0x7c,0x9f,0x5a,0x72,0x80]
          vcvtph2pd xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}

// CHECK: vcvtph2pd ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x2f,0x5a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2pd ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2pd ymm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7c,0x38,0x5a,0x31]
          vcvtph2pd ymm30, word ptr [r9]{1to4}

// CHECK: vcvtph2pd ymm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7c,0x28,0x5a,0x71,0x7f]
          vcvtph2pd ymm30, qword ptr [rcx + 1016]

// CHECK: vcvtph2pd ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7c,0xbf,0x5a,0x72,0x80]
          vcvtph2pd ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvtph2psx xmm30, xmm29
// CHECK: encoding: [0x62,0x06,0x7d,0x08,0x13,0xf5]
          vcvtph2psx xmm30, xmm29

// CHECK: vcvtph2psx ymm30, xmm29
// CHECK: encoding: [0x62,0x06,0x7d,0x28,0x13,0xf5]
          vcvtph2psx ymm30, xmm29

// CHECK: vcvtph2psx xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x0f,0x13,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2psx xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2psx xmm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x46,0x7d,0x18,0x13,0x31]
          vcvtph2psx xmm30, word ptr [r9]{1to4}

// CHECK: vcvtph2psx xmm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x66,0x7d,0x08,0x13,0x71,0x7f]
          vcvtph2psx xmm30, qword ptr [rcx + 1016]

// CHECK: vcvtph2psx xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x66,0x7d,0x9f,0x13,0x72,0x80]
          vcvtph2psx xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvtph2psx ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x2f,0x13,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2psx ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2psx ymm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x7d,0x38,0x13,0x31]
          vcvtph2psx ymm30, word ptr [r9]{1to8}

// CHECK: vcvtph2psx ymm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x7d,0x28,0x13,0x71,0x7f]
          vcvtph2psx ymm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtph2psx ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x7d,0xbf,0x13,0x72,0x80]
          vcvtph2psx ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtph2qq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x7b,0xf5]
          vcvtph2qq xmm30, xmm29

// CHECK: vcvtph2qq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x7b,0xf5]
          vcvtph2qq ymm30, xmm29

// CHECK: vcvtph2qq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x7b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2qq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2qq xmm30, word ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x7b,0x31]
          vcvtph2qq xmm30, word ptr [r9]{1to2}

// CHECK: vcvtph2qq xmm30, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x7b,0x71,0x7f]
          vcvtph2qq xmm30, dword ptr [rcx + 508]

// CHECK: vcvtph2qq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x7b,0x72,0x80]
          vcvtph2qq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}

// CHECK: vcvtph2qq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x2f,0x7b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2qq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2qq ymm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x7b,0x31]
          vcvtph2qq ymm30, word ptr [r9]{1to4}

// CHECK: vcvtph2qq ymm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x7b,0x71,0x7f]
          vcvtph2qq ymm30, qword ptr [rcx + 1016]

// CHECK: vcvtph2qq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x7b,0x72,0x80]
          vcvtph2qq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvtph2udq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x79,0xf5]
          vcvtph2udq xmm30, xmm29

// CHECK: vcvtph2udq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x28,0x79,0xf5]
          vcvtph2udq ymm30, xmm29

// CHECK: vcvtph2udq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x0f,0x79,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2udq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2udq xmm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7c,0x18,0x79,0x31]
          vcvtph2udq xmm30, word ptr [r9]{1to4}

// CHECK: vcvtph2udq xmm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x79,0x71,0x7f]
          vcvtph2udq xmm30, qword ptr [rcx + 1016]

// CHECK: vcvtph2udq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7c,0x9f,0x79,0x72,0x80]
          vcvtph2udq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvtph2udq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x2f,0x79,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2udq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2udq ymm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7c,0x38,0x79,0x31]
          vcvtph2udq ymm30, word ptr [r9]{1to8}

// CHECK: vcvtph2udq ymm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7c,0x28,0x79,0x71,0x7f]
          vcvtph2udq ymm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtph2udq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7c,0xbf,0x79,0x72,0x80]
          vcvtph2udq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtph2uqq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x79,0xf5]
          vcvtph2uqq xmm30, xmm29

// CHECK: vcvtph2uqq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x79,0xf5]
          vcvtph2uqq ymm30, xmm29

// CHECK: vcvtph2uqq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x79,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2uqq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2uqq xmm30, word ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x79,0x31]
          vcvtph2uqq xmm30, word ptr [r9]{1to2}

// CHECK: vcvtph2uqq xmm30, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x79,0x71,0x7f]
          vcvtph2uqq xmm30, dword ptr [rcx + 508]

// CHECK: vcvtph2uqq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x79,0x72,0x80]
          vcvtph2uqq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}

// CHECK: vcvtph2uqq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x2f,0x79,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2uqq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2uqq ymm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x79,0x31]
          vcvtph2uqq ymm30, word ptr [r9]{1to4}

// CHECK: vcvtph2uqq ymm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x79,0x71,0x7f]
          vcvtph2uqq ymm30, qword ptr [rcx + 1016]

// CHECK: vcvtph2uqq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x79,0x72,0x80]
          vcvtph2uqq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvtph2uw xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x7d,0xf5]
          vcvtph2uw xmm30, xmm29

// CHECK: vcvtph2uw ymm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7c,0x28,0x7d,0xf5]
          vcvtph2uw ymm30, ymm29

// CHECK: vcvtph2uw xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x0f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2uw xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2uw xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7c,0x18,0x7d,0x31]
          vcvtph2uw xmm30, word ptr [r9]{1to8}

// CHECK: vcvtph2uw xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x7d,0x71,0x7f]
          vcvtph2uw xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtph2uw xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7c,0x9f,0x7d,0x72,0x80]
          vcvtph2uw xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtph2uw ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x2f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2uw ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2uw ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x7c,0x38,0x7d,0x31]
          vcvtph2uw ymm30, word ptr [r9]{1to16}

// CHECK: vcvtph2uw ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7c,0x28,0x7d,0x71,0x7f]
          vcvtph2uw ymm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtph2uw ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x7c,0xbf,0x7d,0x72,0x80]
          vcvtph2uw ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtph2w xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x7d,0xf5]
          vcvtph2w xmm30, xmm29

// CHECK: vcvtph2w ymm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x7d,0xf5]
          vcvtph2w ymm30, ymm29

// CHECK: vcvtph2w xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2w xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2w xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x7d,0x31]
          vcvtph2w xmm30, word ptr [r9]{1to8}

// CHECK: vcvtph2w xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x7d,0x71,0x7f]
          vcvtph2w xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtph2w xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x7d,0x72,0x80]
          vcvtph2w xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtph2w ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x2f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2w ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtph2w ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x7d,0x31]
          vcvtph2w ymm30, word ptr [r9]{1to16}

// CHECK: vcvtph2w ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x7d,0x71,0x7f]
          vcvtph2w ymm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtph2w ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x7d,0x72,0x80]
          vcvtph2w ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtps2phx xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x1d,0xf5]
          vcvtps2phx xmm30, xmm29

// CHECK: vcvtps2phx xmm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x1d,0xf5]
          vcvtps2phx xmm30, ymm29

// CHECK: vcvtps2phx xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x1d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2phx xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtps2phx xmm30, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x1d,0x31]
          vcvtps2phx xmm30, dword ptr [r9]{1to4}

// CHECK: vcvtps2phx xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x1d,0x71,0x7f]
          vcvtps2phx xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtps2phx xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x1d,0x72,0x80]
          vcvtps2phx xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtps2phx xmm30, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x1d,0x31]
          vcvtps2phx xmm30, dword ptr [r9]{1to8}

// CHECK: vcvtps2phx xmm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x1d,0x71,0x7f]
          vcvtps2phx xmm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtps2phx xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x1d,0x72,0x80]
          vcvtps2phx xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvtqq2ph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0xfc,0x08,0x5b,0xf5]
          vcvtqq2ph xmm30, xmm29

// CHECK: vcvtqq2ph xmm30, ymm29
// CHECK: encoding: [0x62,0x05,0xfc,0x28,0x5b,0xf5]
          vcvtqq2ph xmm30, ymm29

// CHECK: vcvtqq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0xfc,0x0f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtqq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtqq2ph xmm30, qword ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0xfc,0x18,0x5b,0x31]
          vcvtqq2ph xmm30, qword ptr [r9]{1to2}

// CHECK: vcvtqq2ph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0xfc,0x08,0x5b,0x71,0x7f]
          vcvtqq2ph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0x65,0xfc,0x9f,0x5b,0x72,0x80]
          vcvtqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to2}

// CHECK: vcvtqq2ph xmm30, qword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0xfc,0x38,0x5b,0x31]
          vcvtqq2ph xmm30, qword ptr [r9]{1to4}

// CHECK: vcvtqq2ph xmm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0xfc,0x28,0x5b,0x71,0x7f]
          vcvtqq2ph xmm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0x65,0xfc,0xbf,0x5b,0x72,0x80]
          vcvtqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to4}

// CHECK: vcvttph2dq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7e,0x08,0x5b,0xf5]
          vcvttph2dq xmm30, xmm29

// CHECK: vcvttph2dq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7e,0x28,0x5b,0xf5]
          vcvttph2dq ymm30, xmm29

// CHECK: vcvttph2dq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7e,0x0f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2dq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2dq xmm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7e,0x18,0x5b,0x31]
          vcvttph2dq xmm30, word ptr [r9]{1to4}

// CHECK: vcvttph2dq xmm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7e,0x08,0x5b,0x71,0x7f]
          vcvttph2dq xmm30, qword ptr [rcx + 1016]

// CHECK: vcvttph2dq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7e,0x9f,0x5b,0x72,0x80]
          vcvttph2dq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvttph2dq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7e,0x2f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2dq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2dq ymm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7e,0x38,0x5b,0x31]
          vcvttph2dq ymm30, word ptr [r9]{1to8}

// CHECK: vcvttph2dq ymm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7e,0x28,0x5b,0x71,0x7f]
          vcvttph2dq ymm30, xmmword ptr [rcx + 2032]

// CHECK: vcvttph2dq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7e,0xbf,0x5b,0x72,0x80]
          vcvttph2dq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttph2qq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x7a,0xf5]
          vcvttph2qq xmm30, xmm29

// CHECK: vcvttph2qq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x7a,0xf5]
          vcvttph2qq ymm30, xmm29

// CHECK: vcvttph2qq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x7a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2qq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2qq xmm30, word ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x7a,0x31]
          vcvttph2qq xmm30, word ptr [r9]{1to2}

// CHECK: vcvttph2qq xmm30, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x7a,0x71,0x7f]
          vcvttph2qq xmm30, dword ptr [rcx + 508]

// CHECK: vcvttph2qq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x7a,0x72,0x80]
          vcvttph2qq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}

// CHECK: vcvttph2qq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x2f,0x7a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2qq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2qq ymm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x7a,0x31]
          vcvttph2qq ymm30, word ptr [r9]{1to4}

// CHECK: vcvttph2qq ymm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x7a,0x71,0x7f]
          vcvttph2qq ymm30, qword ptr [rcx + 1016]

// CHECK: vcvttph2qq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x7a,0x72,0x80]
          vcvttph2qq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvttph2udq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x78,0xf5]
          vcvttph2udq xmm30, xmm29

// CHECK: vcvttph2udq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x28,0x78,0xf5]
          vcvttph2udq ymm30, xmm29

// CHECK: vcvttph2udq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x0f,0x78,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2udq xmm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2udq xmm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7c,0x18,0x78,0x31]
          vcvttph2udq xmm30, word ptr [r9]{1to4}

// CHECK: vcvttph2udq xmm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x78,0x71,0x7f]
          vcvttph2udq xmm30, qword ptr [rcx + 1016]

// CHECK: vcvttph2udq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7c,0x9f,0x78,0x72,0x80]
          vcvttph2udq xmm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvttph2udq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x2f,0x78,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2udq ymm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2udq ymm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7c,0x38,0x78,0x31]
          vcvttph2udq ymm30, word ptr [r9]{1to8}

// CHECK: vcvttph2udq ymm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7c,0x28,0x78,0x71,0x7f]
          vcvttph2udq ymm30, xmmword ptr [rcx + 2032]

// CHECK: vcvttph2udq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7c,0xbf,0x78,0x72,0x80]
          vcvttph2udq ymm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttph2uqq xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x78,0xf5]
          vcvttph2uqq xmm30, xmm29

// CHECK: vcvttph2uqq ymm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x78,0xf5]
          vcvttph2uqq ymm30, xmm29

// CHECK: vcvttph2uqq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x78,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2uqq xmm30 {k7}, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2uqq xmm30, word ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x78,0x31]
          vcvttph2uqq xmm30, word ptr [r9]{1to2}

// CHECK: vcvttph2uqq xmm30, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x78,0x71,0x7f]
          vcvttph2uqq xmm30, dword ptr [rcx + 508]

// CHECK: vcvttph2uqq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x78,0x72,0x80]
          vcvttph2uqq xmm30 {k7} {z}, word ptr [rdx - 256]{1to2}

// CHECK: vcvttph2uqq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x2f,0x78,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2uqq ymm30 {k7}, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2uqq ymm30, word ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x78,0x31]
          vcvttph2uqq ymm30, word ptr [r9]{1to4}

// CHECK: vcvttph2uqq ymm30, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x78,0x71,0x7f]
          vcvttph2uqq ymm30, qword ptr [rcx + 1016]

// CHECK: vcvttph2uqq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x78,0x72,0x80]
          vcvttph2uqq ymm30 {k7} {z}, word ptr [rdx - 256]{1to4}

// CHECK: vcvttph2uw xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x7c,0xf5]
          vcvttph2uw xmm30, xmm29

// CHECK: vcvttph2uw ymm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7c,0x28,0x7c,0xf5]
          vcvttph2uw ymm30, ymm29

// CHECK: vcvttph2uw xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x0f,0x7c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2uw xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2uw xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7c,0x18,0x7c,0x31]
          vcvttph2uw xmm30, word ptr [r9]{1to8}

// CHECK: vcvttph2uw xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x7c,0x71,0x7f]
          vcvttph2uw xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvttph2uw xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7c,0x9f,0x7c,0x72,0x80]
          vcvttph2uw xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttph2uw ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x2f,0x7c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2uw ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2uw ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x7c,0x38,0x7c,0x31]
          vcvttph2uw ymm30, word ptr [r9]{1to16}

// CHECK: vcvttph2uw ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7c,0x28,0x7c,0x71,0x7f]
          vcvttph2uw ymm30, ymmword ptr [rcx + 4064]

// CHECK: vcvttph2uw ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x7c,0xbf,0x7c,0x72,0x80]
          vcvttph2uw ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvttph2w xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7d,0x08,0x7c,0xf5]
          vcvttph2w xmm30, xmm29

// CHECK: vcvttph2w ymm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7d,0x28,0x7c,0xf5]
          vcvttph2w ymm30, ymm29

// CHECK: vcvttph2w xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x0f,0x7c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2w xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2w xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7d,0x18,0x7c,0x31]
          vcvttph2w xmm30, word ptr [r9]{1to8}

// CHECK: vcvttph2w xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x7c,0x71,0x7f]
          vcvttph2w xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvttph2w xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7d,0x9f,0x7c,0x72,0x80]
          vcvttph2w xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvttph2w ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7d,0x2f,0x7c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2w ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttph2w ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x7d,0x38,0x7c,0x31]
          vcvttph2w ymm30, word ptr [r9]{1to16}

// CHECK: vcvttph2w ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7d,0x28,0x7c,0x71,0x7f]
          vcvttph2w ymm30, ymmword ptr [rcx + 4064]

// CHECK: vcvttph2w ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x7d,0xbf,0x7c,0x72,0x80]
          vcvttph2w ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtudq2ph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7f,0x08,0x7a,0xf5]
          vcvtudq2ph xmm30, xmm29

// CHECK: vcvtudq2ph xmm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7f,0x28,0x7a,0xf5]
          vcvtudq2ph xmm30, ymm29

// CHECK: vcvtudq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7f,0x0f,0x7a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtudq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtudq2ph xmm30, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0x7f,0x18,0x7a,0x31]
          vcvtudq2ph xmm30, dword ptr [r9]{1to4}

// CHECK: vcvtudq2ph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7f,0x08,0x7a,0x71,0x7f]
          vcvtudq2ph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtudq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x65,0x7f,0x9f,0x7a,0x72,0x80]
          vcvtudq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvtudq2ph xmm30, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7f,0x38,0x7a,0x31]
          vcvtudq2ph xmm30, dword ptr [r9]{1to8}

// CHECK: vcvtudq2ph xmm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7f,0x28,0x7a,0x71,0x7f]
          vcvtudq2ph xmm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtudq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x65,0x7f,0xbf,0x7a,0x72,0x80]
          vcvtudq2ph xmm30 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvtuqq2ph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0xff,0x08,0x7a,0xf5]
          vcvtuqq2ph xmm30, xmm29

// CHECK: vcvtuqq2ph xmm30, ymm29
// CHECK: encoding: [0x62,0x05,0xff,0x28,0x7a,0xf5]
          vcvtuqq2ph xmm30, ymm29

// CHECK: vcvtuqq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0xff,0x0f,0x7a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtuqq2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtuqq2ph xmm30, qword ptr [r9]{1to2}
// CHECK: encoding: [0x62,0x45,0xff,0x18,0x7a,0x31]
          vcvtuqq2ph xmm30, qword ptr [r9]{1to2}

// CHECK: vcvtuqq2ph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0xff,0x08,0x7a,0x71,0x7f]
          vcvtuqq2ph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtuqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0x65,0xff,0x9f,0x7a,0x72,0x80]
          vcvtuqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to2}

// CHECK: vcvtuqq2ph xmm30, qword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x45,0xff,0x38,0x7a,0x31]
          vcvtuqq2ph xmm30, qword ptr [r9]{1to4}

// CHECK: vcvtuqq2ph xmm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0xff,0x28,0x7a,0x71,0x7f]
          vcvtuqq2ph xmm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtuqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0x65,0xff,0xbf,0x7a,0x72,0x80]
          vcvtuqq2ph xmm30 {k7} {z}, qword ptr [rdx - 1024]{1to4}

// CHECK: vcvtuw2ph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7f,0x08,0x7d,0xf5]
          vcvtuw2ph xmm30, xmm29

// CHECK: vcvtuw2ph ymm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7f,0x28,0x7d,0xf5]
          vcvtuw2ph ymm30, ymm29

// CHECK: vcvtuw2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7f,0x0f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtuw2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtuw2ph xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7f,0x18,0x7d,0x31]
          vcvtuw2ph xmm30, word ptr [r9]{1to8}

// CHECK: vcvtuw2ph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7f,0x08,0x7d,0x71,0x7f]
          vcvtuw2ph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtuw2ph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7f,0x9f,0x7d,0x72,0x80]
          vcvtuw2ph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtuw2ph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7f,0x2f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtuw2ph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtuw2ph ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x7f,0x38,0x7d,0x31]
          vcvtuw2ph ymm30, word ptr [r9]{1to16}

// CHECK: vcvtuw2ph ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7f,0x28,0x7d,0x71,0x7f]
          vcvtuw2ph ymm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtuw2ph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x7f,0xbf,0x7d,0x72,0x80]
          vcvtuw2ph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vcvtw2ph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7e,0x08,0x7d,0xf5]
          vcvtw2ph xmm30, xmm29

// CHECK: vcvtw2ph ymm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7e,0x28,0x7d,0xf5]
          vcvtw2ph ymm30, ymm29

// CHECK: vcvtw2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7e,0x0f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtw2ph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtw2ph xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7e,0x18,0x7d,0x31]
          vcvtw2ph xmm30, word ptr [r9]{1to8}

// CHECK: vcvtw2ph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7e,0x08,0x7d,0x71,0x7f]
          vcvtw2ph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vcvtw2ph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7e,0x9f,0x7d,0x72,0x80]
          vcvtw2ph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vcvtw2ph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7e,0x2f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtw2ph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvtw2ph ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x7e,0x38,0x7d,0x31]
          vcvtw2ph ymm30, word ptr [r9]{1to16}

// CHECK: vcvtw2ph ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7e,0x28,0x7d,0x71,0x7f]
          vcvtw2ph ymm30, ymmword ptr [rcx + 4064]

// CHECK: vcvtw2ph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x7e,0xbf,0x7d,0x72,0x80]
          vcvtw2ph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vfpclassph k5, xmm30, 123
// CHECK: encoding: [0x62,0x93,0x7c,0x08,0x66,0xee,0x7b]
          vfpclassph k5, xmm30, 123

// CHECK: vfpclassph k5, ymm30, 123
// CHECK: encoding: [0x62,0x93,0x7c,0x28,0x66,0xee,0x7b]
          vfpclassph k5, ymm30, 123

// CHECK: vfpclassph k5 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0xb3,0x7c,0x0f,0x66,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vfpclassph k5 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vfpclassph k5, word ptr [r9]{1to8}, 123
// CHECK: encoding: [0x62,0xd3,0x7c,0x18,0x66,0x29,0x7b]
          vfpclassph k5, word ptr [r9]{1to8}, 123

// CHECK: vfpclassph k5, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x66,0x69,0x7f,0x7b]
          vfpclassph k5, xmmword ptr [rcx + 2032], 123

// CHECK: vfpclassph k5 {k7}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclassph k5 {k7}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vfpclassph k5, word ptr [r9]{1to16}, 123
// CHECK: encoding: [0x62,0xd3,0x7c,0x38,0x66,0x29,0x7b]
          vfpclassph k5, word ptr [r9]{1to16}, 123

// CHECK: vfpclassph k5, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x66,0x69,0x7f,0x7b]
          vfpclassph k5, ymmword ptr [rcx + 4064], 123

// CHECK: vfpclassph k5 {k7}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0xf3,0x7c,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclassph k5 {k7}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vgetexpph xmm30, xmm29
// CHECK: encoding: [0x62,0x06,0x7d,0x08,0x42,0xf5]
          vgetexpph xmm30, xmm29

// CHECK: vgetexpph ymm30, ymm29
// CHECK: encoding: [0x62,0x06,0x7d,0x28,0x42,0xf5]
          vgetexpph ymm30, ymm29

// CHECK: vgetexpph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x0f,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexpph xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x7d,0x18,0x42,0x31]
          vgetexpph xmm30, word ptr [r9]{1to8}

// CHECK: vgetexpph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x7d,0x08,0x42,0x71,0x7f]
          vgetexpph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vgetexpph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x7d,0x9f,0x42,0x72,0x80]
          vgetexpph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vgetexpph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x2f,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vgetexpph ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x7d,0x38,0x42,0x31]
          vgetexpph ymm30, word ptr [r9]{1to16}

// CHECK: vgetexpph ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x7d,0x28,0x42,0x71,0x7f]
          vgetexpph ymm30, ymmword ptr [rcx + 4064]

// CHECK: vgetexpph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x7d,0xbf,0x42,0x72,0x80]
          vgetexpph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vgetmantph ymm30, ymm29, 123
// CHECK: encoding: [0x62,0x03,0x7c,0x28,0x26,0xf5,0x7b]
          vgetmantph ymm30, ymm29, 123

// CHECK: vgetmantph xmm30, xmm29, 123
// CHECK: encoding: [0x62,0x03,0x7c,0x08,0x26,0xf5,0x7b]
          vgetmantph xmm30, xmm29, 123

// CHECK: vgetmantph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0x23,0x7c,0x0f,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantph xmm30, word ptr [r9]{1to8}, 123
// CHECK: encoding: [0x62,0x43,0x7c,0x18,0x26,0x31,0x7b]
          vgetmantph xmm30, word ptr [r9]{1to8}, 123

// CHECK: vgetmantph xmm30, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0x63,0x7c,0x08,0x26,0x71,0x7f,0x7b]
          vgetmantph xmm30, xmmword ptr [rcx + 2032], 123

// CHECK: vgetmantph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0x63,0x7c,0x9f,0x26,0x72,0x80,0x7b]
          vgetmantph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vgetmantph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0x23,0x7c,0x2f,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vgetmantph ymm30, word ptr [r9]{1to16}, 123
// CHECK: encoding: [0x62,0x43,0x7c,0x38,0x26,0x31,0x7b]
          vgetmantph ymm30, word ptr [r9]{1to16}, 123

// CHECK: vgetmantph ymm30, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0x63,0x7c,0x28,0x26,0x71,0x7f,0x7b]
          vgetmantph ymm30, ymmword ptr [rcx + 4064], 123

// CHECK: vgetmantph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0x63,0x7c,0xbf,0x26,0x72,0x80,0x7b]
          vgetmantph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vrcpph xmm30, xmm29
// CHECK: encoding: [0x62,0x06,0x7d,0x08,0x4c,0xf5]
          vrcpph xmm30, xmm29

// CHECK: vrcpph ymm30, ymm29
// CHECK: encoding: [0x62,0x06,0x7d,0x28,0x4c,0xf5]
          vrcpph ymm30, ymm29

// CHECK: vrcpph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x0f,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcpph xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x7d,0x18,0x4c,0x31]
          vrcpph xmm30, word ptr [r9]{1to8}

// CHECK: vrcpph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x7d,0x08,0x4c,0x71,0x7f]
          vrcpph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vrcpph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x7d,0x9f,0x4c,0x72,0x80]
          vrcpph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vrcpph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x2f,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrcpph ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x7d,0x38,0x4c,0x31]
          vrcpph ymm30, word ptr [r9]{1to16}

// CHECK: vrcpph ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x7d,0x28,0x4c,0x71,0x7f]
          vrcpph ymm30, ymmword ptr [rcx + 4064]

// CHECK: vrcpph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x7d,0xbf,0x4c,0x72,0x80]
          vrcpph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vreduceph ymm30, ymm29, 123
// CHECK: encoding: [0x62,0x03,0x7c,0x28,0x56,0xf5,0x7b]
          vreduceph ymm30, ymm29, 123

// CHECK: vreduceph xmm30, xmm29, 123
// CHECK: encoding: [0x62,0x03,0x7c,0x08,0x56,0xf5,0x7b]
          vreduceph xmm30, xmm29, 123

// CHECK: vreduceph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0x23,0x7c,0x0f,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreduceph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreduceph xmm30, word ptr [r9]{1to8}, 123
// CHECK: encoding: [0x62,0x43,0x7c,0x18,0x56,0x31,0x7b]
          vreduceph xmm30, word ptr [r9]{1to8}, 123

// CHECK: vreduceph xmm30, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0x63,0x7c,0x08,0x56,0x71,0x7f,0x7b]
          vreduceph xmm30, xmmword ptr [rcx + 2032], 123

// CHECK: vreduceph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0x63,0x7c,0x9f,0x56,0x72,0x80,0x7b]
          vreduceph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vreduceph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0x23,0x7c,0x2f,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreduceph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vreduceph ymm30, word ptr [r9]{1to16}, 123
// CHECK: encoding: [0x62,0x43,0x7c,0x38,0x56,0x31,0x7b]
          vreduceph ymm30, word ptr [r9]{1to16}, 123

// CHECK: vreduceph ymm30, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0x63,0x7c,0x28,0x56,0x71,0x7f,0x7b]
          vreduceph ymm30, ymmword ptr [rcx + 4064], 123

// CHECK: vreduceph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0x63,0x7c,0xbf,0x56,0x72,0x80,0x7b]
          vreduceph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vrndscaleph ymm30, ymm29, 123
// CHECK: encoding: [0x62,0x03,0x7c,0x28,0x08,0xf5,0x7b]
          vrndscaleph ymm30, ymm29, 123

// CHECK: vrndscaleph xmm30, xmm29, 123
// CHECK: encoding: [0x62,0x03,0x7c,0x08,0x08,0xf5,0x7b]
          vrndscaleph xmm30, xmm29, 123

// CHECK: vrndscaleph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0x23,0x7c,0x0f,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscaleph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscaleph xmm30, word ptr [r9]{1to8}, 123
// CHECK: encoding: [0x62,0x43,0x7c,0x18,0x08,0x31,0x7b]
          vrndscaleph xmm30, word ptr [r9]{1to8}, 123

// CHECK: vrndscaleph xmm30, xmmword ptr [rcx + 2032], 123
// CHECK: encoding: [0x62,0x63,0x7c,0x08,0x08,0x71,0x7f,0x7b]
          vrndscaleph xmm30, xmmword ptr [rcx + 2032], 123

// CHECK: vrndscaleph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}, 123
// CHECK: encoding: [0x62,0x63,0x7c,0x9f,0x08,0x72,0x80,0x7b]
          vrndscaleph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}, 123

// CHECK: vrndscaleph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456], 123
// CHECK: encoding: [0x62,0x23,0x7c,0x2f,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscaleph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456], 123

// CHECK: vrndscaleph ymm30, word ptr [r9]{1to16}, 123
// CHECK: encoding: [0x62,0x43,0x7c,0x38,0x08,0x31,0x7b]
          vrndscaleph ymm30, word ptr [r9]{1to16}, 123

// CHECK: vrndscaleph ymm30, ymmword ptr [rcx + 4064], 123
// CHECK: encoding: [0x62,0x63,0x7c,0x28,0x08,0x71,0x7f,0x7b]
          vrndscaleph ymm30, ymmword ptr [rcx + 4064], 123

// CHECK: vrndscaleph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}, 123
// CHECK: encoding: [0x62,0x63,0x7c,0xbf,0x08,0x72,0x80,0x7b]
          vrndscaleph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}, 123

// CHECK: vrsqrtph xmm30, xmm29
// CHECK: encoding: [0x62,0x06,0x7d,0x08,0x4e,0xf5]
          vrsqrtph xmm30, xmm29

// CHECK: vrsqrtph ymm30, ymm29
// CHECK: encoding: [0x62,0x06,0x7d,0x28,0x4e,0xf5]
          vrsqrtph ymm30, ymm29

// CHECK: vrsqrtph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x0f,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtph xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x7d,0x18,0x4e,0x31]
          vrsqrtph xmm30, word ptr [r9]{1to8}

// CHECK: vrsqrtph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x7d,0x08,0x4e,0x71,0x7f]
          vrsqrtph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vrsqrtph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x7d,0x9f,0x4e,0x72,0x80]
          vrsqrtph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vrsqrtph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x7d,0x2f,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vrsqrtph ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x7d,0x38,0x4e,0x31]
          vrsqrtph ymm30, word ptr [r9]{1to16}

// CHECK: vrsqrtph ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x7d,0x28,0x4e,0x71,0x7f]
          vrsqrtph ymm30, ymmword ptr [rcx + 4064]

// CHECK: vrsqrtph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x7d,0xbf,0x4e,0x72,0x80]
          vrsqrtph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vscalefph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0x2c,0xf4]
          vscalefph ymm30, ymm29, ymm28

// CHECK: vscalefph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x2c,0xf4]
          vscalefph xmm30, xmm29, xmm28

// CHECK: vscalefph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0x2c,0x31]
          vscalefph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vscalefph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0x2c,0x71,0x7f]
          vscalefph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vscalefph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0x2c,0x72,0x80]
          vscalefph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vscalefph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vscalefph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0x2c,0x31]
          vscalefph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vscalefph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x2c,0x71,0x7f]
          vscalefph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vscalefph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0x2c,0x72,0x80]
          vscalefph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vsqrtph xmm30, xmm29
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x51,0xf5]
          vsqrtph xmm30, xmm29

// CHECK: vsqrtph ymm30, ymm29
// CHECK: encoding: [0x62,0x05,0x7c,0x28,0x51,0xf5]
          vsqrtph ymm30, ymm29

// CHECK: vsqrtph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x0f,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtph xmm30 {k7}, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtph xmm30, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x45,0x7c,0x18,0x51,0x31]
          vsqrtph xmm30, word ptr [r9]{1to8}

// CHECK: vsqrtph xmm30, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x51,0x71,0x7f]
          vsqrtph xmm30, xmmword ptr [rcx + 2032]

// CHECK: vsqrtph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x65,0x7c,0x9f,0x51,0x72,0x80]
          vsqrtph xmm30 {k7} {z}, word ptr [rdx - 256]{1to8}

// CHECK: vsqrtph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x25,0x7c,0x2f,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtph ymm30 {k7}, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vsqrtph ymm30, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x45,0x7c,0x38,0x51,0x31]
          vsqrtph ymm30, word ptr [r9]{1to16}

// CHECK: vsqrtph ymm30, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x65,0x7c,0x28,0x51,0x71,0x7f]
          vsqrtph ymm30, ymmword ptr [rcx + 4064]

// CHECK: vsqrtph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x65,0x7c,0xbf,0x51,0x72,0x80]
          vsqrtph ymm30 {k7} {z}, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd132ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0x98,0xf4]
          vfmadd132ph ymm30, ymm29, ymm28

// CHECK: vfmadd132ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x98,0xf4]
          vfmadd132ph xmm30, xmm29, xmm28

// CHECK: vfmadd132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0x98,0x31]
          vfmadd132ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmadd132ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0x98,0x71,0x7f]
          vfmadd132ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmadd132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0x98,0x72,0x80]
          vfmadd132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd132ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0x98,0x31]
          vfmadd132ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmadd132ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x98,0x71,0x7f]
          vfmadd132ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmadd132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0x98,0x72,0x80]
          vfmadd132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd213ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xa8,0xf4]
          vfmadd213ph ymm30, ymm29, ymm28

// CHECK: vfmadd213ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xa8,0xf4]
          vfmadd213ph xmm30, xmm29, xmm28

// CHECK: vfmadd213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xa8,0x31]
          vfmadd213ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmadd213ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xa8,0x71,0x7f]
          vfmadd213ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmadd213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xa8,0x72,0x80]
          vfmadd213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd213ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xa8,0x31]
          vfmadd213ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmadd213ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xa8,0x71,0x7f]
          vfmadd213ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmadd213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xa8,0x72,0x80]
          vfmadd213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmadd231ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xb8,0xf4]
          vfmadd231ph ymm30, ymm29, ymm28

// CHECK: vfmadd231ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xb8,0xf4]
          vfmadd231ph xmm30, xmm29, xmm28

// CHECK: vfmadd231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xb8,0x31]
          vfmadd231ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmadd231ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xb8,0x71,0x7f]
          vfmadd231ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmadd231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xb8,0x72,0x80]
          vfmadd231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmadd231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmadd231ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xb8,0x31]
          vfmadd231ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmadd231ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xb8,0x71,0x7f]
          vfmadd231ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmadd231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xb8,0x72,0x80]
          vfmadd231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmaddsub132ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0x96,0xf4]
          vfmaddsub132ph ymm30, ymm29, ymm28

// CHECK: vfmaddsub132ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x96,0xf4]
          vfmaddsub132ph xmm30, xmm29, xmm28

// CHECK: vfmaddsub132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0x96,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddsub132ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0x96,0x31]
          vfmaddsub132ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmaddsub132ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0x96,0x71,0x7f]
          vfmaddsub132ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmaddsub132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0x96,0x72,0x80]
          vfmaddsub132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmaddsub132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x96,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddsub132ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0x96,0x31]
          vfmaddsub132ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmaddsub132ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x96,0x71,0x7f]
          vfmaddsub132ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmaddsub132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0x96,0x72,0x80]
          vfmaddsub132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmaddsub213ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xa6,0xf4]
          vfmaddsub213ph ymm30, ymm29, ymm28

// CHECK: vfmaddsub213ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xa6,0xf4]
          vfmaddsub213ph xmm30, xmm29, xmm28

// CHECK: vfmaddsub213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xa6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddsub213ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xa6,0x31]
          vfmaddsub213ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmaddsub213ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xa6,0x71,0x7f]
          vfmaddsub213ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmaddsub213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xa6,0x72,0x80]
          vfmaddsub213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmaddsub213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xa6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddsub213ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xa6,0x31]
          vfmaddsub213ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmaddsub213ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xa6,0x71,0x7f]
          vfmaddsub213ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmaddsub213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xa6,0x72,0x80]
          vfmaddsub213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmaddsub231ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xb6,0xf4]
          vfmaddsub231ph ymm30, ymm29, ymm28

// CHECK: vfmaddsub231ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xb6,0xf4]
          vfmaddsub231ph xmm30, xmm29, xmm28

// CHECK: vfmaddsub231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xb6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddsub231ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xb6,0x31]
          vfmaddsub231ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmaddsub231ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xb6,0x71,0x7f]
          vfmaddsub231ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmaddsub231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xb6,0x72,0x80]
          vfmaddsub231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmaddsub231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xb6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddsub231ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xb6,0x31]
          vfmaddsub231ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmaddsub231ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xb6,0x71,0x7f]
          vfmaddsub231ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmaddsub231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xb6,0x72,0x80]
          vfmaddsub231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub132ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0x9a,0xf4]
          vfmsub132ph ymm30, ymm29, ymm28

// CHECK: vfmsub132ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x9a,0xf4]
          vfmsub132ph xmm30, xmm29, xmm28

// CHECK: vfmsub132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0x9a,0x31]
          vfmsub132ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmsub132ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0x9a,0x71,0x7f]
          vfmsub132ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmsub132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0x9a,0x72,0x80]
          vfmsub132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub132ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0x9a,0x31]
          vfmsub132ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmsub132ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x9a,0x71,0x7f]
          vfmsub132ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmsub132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0x9a,0x72,0x80]
          vfmsub132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub213ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xaa,0xf4]
          vfmsub213ph ymm30, ymm29, ymm28

// CHECK: vfmsub213ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xaa,0xf4]
          vfmsub213ph xmm30, xmm29, xmm28

// CHECK: vfmsub213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xaa,0x31]
          vfmsub213ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmsub213ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xaa,0x71,0x7f]
          vfmsub213ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmsub213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xaa,0x72,0x80]
          vfmsub213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub213ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xaa,0x31]
          vfmsub213ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmsub213ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xaa,0x71,0x7f]
          vfmsub213ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmsub213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xaa,0x72,0x80]
          vfmsub213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmsub231ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xba,0xf4]
          vfmsub231ph ymm30, ymm29, ymm28

// CHECK: vfmsub231ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xba,0xf4]
          vfmsub231ph xmm30, xmm29, xmm28

// CHECK: vfmsub231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xba,0x31]
          vfmsub231ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmsub231ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xba,0x71,0x7f]
          vfmsub231ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmsub231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xba,0x72,0x80]
          vfmsub231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmsub231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsub231ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xba,0x31]
          vfmsub231ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmsub231ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xba,0x71,0x7f]
          vfmsub231ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmsub231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xba,0x72,0x80]
          vfmsub231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmsubadd132ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0x97,0xf4]
          vfmsubadd132ph ymm30, ymm29, ymm28

// CHECK: vfmsubadd132ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x97,0xf4]
          vfmsubadd132ph xmm30, xmm29, xmm28

// CHECK: vfmsubadd132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0x97,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsubadd132ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0x97,0x31]
          vfmsubadd132ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmsubadd132ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0x97,0x71,0x7f]
          vfmsubadd132ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmsubadd132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0x97,0x72,0x80]
          vfmsubadd132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmsubadd132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x97,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsubadd132ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0x97,0x31]
          vfmsubadd132ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmsubadd132ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x97,0x71,0x7f]
          vfmsubadd132ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmsubadd132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0x97,0x72,0x80]
          vfmsubadd132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmsubadd213ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xa7,0xf4]
          vfmsubadd213ph ymm30, ymm29, ymm28

// CHECK: vfmsubadd213ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xa7,0xf4]
          vfmsubadd213ph xmm30, xmm29, xmm28

// CHECK: vfmsubadd213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xa7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsubadd213ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xa7,0x31]
          vfmsubadd213ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmsubadd213ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xa7,0x71,0x7f]
          vfmsubadd213ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmsubadd213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xa7,0x72,0x80]
          vfmsubadd213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmsubadd213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xa7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsubadd213ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xa7,0x31]
          vfmsubadd213ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmsubadd213ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xa7,0x71,0x7f]
          vfmsubadd213ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmsubadd213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xa7,0x72,0x80]
          vfmsubadd213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfmsubadd231ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xb7,0xf4]
          vfmsubadd231ph ymm30, ymm29, ymm28

// CHECK: vfmsubadd231ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xb7,0xf4]
          vfmsubadd231ph xmm30, xmm29, xmm28

// CHECK: vfmsubadd231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xb7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsubadd231ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xb7,0x31]
          vfmsubadd231ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfmsubadd231ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xb7,0x71,0x7f]
          vfmsubadd231ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmsubadd231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xb7,0x72,0x80]
          vfmsubadd231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfmsubadd231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xb7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmsubadd231ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xb7,0x31]
          vfmsubadd231ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfmsubadd231ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xb7,0x71,0x7f]
          vfmsubadd231ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmsubadd231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xb7,0x72,0x80]
          vfmsubadd231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd132ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0x9c,0xf4]
          vfnmadd132ph ymm30, ymm29, ymm28

// CHECK: vfnmadd132ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x9c,0xf4]
          vfnmadd132ph xmm30, xmm29, xmm28

// CHECK: vfnmadd132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0x9c,0x31]
          vfnmadd132ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfnmadd132ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0x9c,0x71,0x7f]
          vfnmadd132ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0x9c,0x72,0x80]
          vfnmadd132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd132ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0x9c,0x31]
          vfnmadd132ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfnmadd132ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x9c,0x71,0x7f]
          vfnmadd132ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0x9c,0x72,0x80]
          vfnmadd132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd213ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xac,0xf4]
          vfnmadd213ph ymm30, ymm29, ymm28

// CHECK: vfnmadd213ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xac,0xf4]
          vfnmadd213ph xmm30, xmm29, xmm28

// CHECK: vfnmadd213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xac,0x31]
          vfnmadd213ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfnmadd213ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xac,0x71,0x7f]
          vfnmadd213ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xac,0x72,0x80]
          vfnmadd213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd213ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xac,0x31]
          vfnmadd213ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfnmadd213ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xac,0x71,0x7f]
          vfnmadd213ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xac,0x72,0x80]
          vfnmadd213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfnmadd231ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xbc,0xf4]
          vfnmadd231ph ymm30, ymm29, ymm28

// CHECK: vfnmadd231ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xbc,0xf4]
          vfnmadd231ph xmm30, xmm29, xmm28

// CHECK: vfnmadd231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xbc,0x31]
          vfnmadd231ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfnmadd231ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xbc,0x71,0x7f]
          vfnmadd231ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfnmadd231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xbc,0x72,0x80]
          vfnmadd231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfnmadd231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmadd231ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xbc,0x31]
          vfnmadd231ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfnmadd231ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xbc,0x71,0x7f]
          vfnmadd231ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfnmadd231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xbc,0x72,0x80]
          vfnmadd231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub132ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0x9e,0xf4]
          vfnmsub132ph ymm30, ymm29, ymm28

// CHECK: vfnmsub132ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x9e,0xf4]
          vfnmsub132ph xmm30, xmm29, xmm28

// CHECK: vfnmsub132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0x9e,0x31]
          vfnmsub132ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfnmsub132ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0x9e,0x71,0x7f]
          vfnmsub132ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0x9e,0x72,0x80]
          vfnmsub132ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub132ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0x9e,0x31]
          vfnmsub132ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfnmsub132ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x9e,0x71,0x7f]
          vfnmsub132ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0x9e,0x72,0x80]
          vfnmsub132ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub213ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xae,0xf4]
          vfnmsub213ph ymm30, ymm29, ymm28

// CHECK: vfnmsub213ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xae,0xf4]
          vfnmsub213ph xmm30, xmm29, xmm28

// CHECK: vfnmsub213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xae,0x31]
          vfnmsub213ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfnmsub213ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xae,0x71,0x7f]
          vfnmsub213ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xae,0x72,0x80]
          vfnmsub213ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub213ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xae,0x31]
          vfnmsub213ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfnmsub213ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xae,0x71,0x7f]
          vfnmsub213ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xae,0x72,0x80]
          vfnmsub213ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfnmsub231ph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x15,0x20,0xbe,0xf4]
          vfnmsub231ph ymm30, ymm29, ymm28

// CHECK: vfnmsub231ph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xbe,0xf4]
          vfnmsub231ph xmm30, xmm29, xmm28

// CHECK: vfnmsub231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x27,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231ph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231ph ymm30, ymm29, word ptr [r9]{1to16}
// CHECK: encoding: [0x62,0x46,0x15,0x30,0xbe,0x31]
          vfnmsub231ph ymm30, ymm29, word ptr [r9]{1to16}

// CHECK: vfnmsub231ph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x15,0x20,0xbe,0x71,0x7f]
          vfnmsub231ph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfnmsub231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}
// CHECK: encoding: [0x62,0x66,0x15,0xb7,0xbe,0x72,0x80]
          vfnmsub231ph ymm30 {k7} {z}, ymm29, word ptr [rdx - 256]{1to16}

// CHECK: vfnmsub231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231ph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfnmsub231ph xmm30, xmm29, word ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x15,0x10,0xbe,0x31]
          vfnmsub231ph xmm30, xmm29, word ptr [r9]{1to8}

// CHECK: vfnmsub231ph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xbe,0x71,0x7f]
          vfnmsub231ph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfnmsub231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}
// CHECK: encoding: [0x62,0x66,0x15,0x97,0xbe,0x72,0x80]
          vfnmsub231ph xmm30 {k7} {z}, xmm29, word ptr [rdx - 256]{1to8}

// CHECK: vfcmaddcph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x17,0x20,0x56,0xf4]
          vfcmaddcph ymm30, ymm29, ymm28

// CHECK: vfcmaddcph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x17,0x00,0x56,0xf4]
          vfcmaddcph xmm30, xmm29, xmm28

// CHECK: vfcmaddcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x17,0x27,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmaddcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfcmaddcph ymm30, ymm29, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x17,0x30,0x56,0x31]
          vfcmaddcph ymm30, ymm29, dword ptr [r9]{1to8}

// CHECK: vfcmaddcph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x17,0x20,0x56,0x71,0x7f]
          vfcmaddcph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfcmaddcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x66,0x17,0xb7,0x56,0x72,0x80]
          vfcmaddcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}

// CHECK: vfcmaddcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x17,0x07,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmaddcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfcmaddcph xmm30, xmm29, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x46,0x17,0x10,0x56,0x31]
          vfcmaddcph xmm30, xmm29, dword ptr [r9]{1to4}

// CHECK: vfcmaddcph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x17,0x00,0x56,0x71,0x7f]
          vfcmaddcph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfcmaddcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x66,0x17,0x97,0x56,0x72,0x80]
          vfcmaddcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}

// CHECK: vfcmulcph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x17,0x20,0xd6,0xf4]
          vfcmulcph ymm30, ymm29, ymm28

// CHECK: vfcmulcph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x17,0x00,0xd6,0xf4]
          vfcmulcph xmm30, xmm29, xmm28

// CHECK: vfcmulcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x17,0x27,0xd6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmulcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfcmulcph ymm30, ymm29, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x17,0x30,0xd6,0x31]
          vfcmulcph ymm30, ymm29, dword ptr [r9]{1to8}

// CHECK: vfcmulcph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x17,0x20,0xd6,0x71,0x7f]
          vfcmulcph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfcmulcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x66,0x17,0xb7,0xd6,0x72,0x80]
          vfcmulcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}

// CHECK: vfcmulcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x17,0x07,0xd6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmulcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfcmulcph xmm30, xmm29, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x46,0x17,0x10,0xd6,0x31]
          vfcmulcph xmm30, xmm29, dword ptr [r9]{1to4}

// CHECK: vfcmulcph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x17,0x00,0xd6,0x71,0x7f]
          vfcmulcph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfcmulcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x66,0x17,0x97,0xd6,0x72,0x80]
          vfcmulcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}

// CHECK: vfmaddcph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x16,0x20,0x56,0xf4]
          vfmaddcph ymm30, ymm29, ymm28

// CHECK: vfmaddcph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x16,0x00,0x56,0xf4]
          vfmaddcph xmm30, xmm29, xmm28

// CHECK: vfmaddcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x16,0x27,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddcph ymm30, ymm29, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x16,0x30,0x56,0x31]
          vfmaddcph ymm30, ymm29, dword ptr [r9]{1to8}

// CHECK: vfmaddcph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x16,0x20,0x56,0x71,0x7f]
          vfmaddcph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmaddcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x66,0x16,0xb7,0x56,0x72,0x80]
          vfmaddcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}

// CHECK: vfmaddcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x16,0x07,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmaddcph xmm30, xmm29, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x46,0x16,0x10,0x56,0x31]
          vfmaddcph xmm30, xmm29, dword ptr [r9]{1to4}

// CHECK: vfmaddcph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x16,0x00,0x56,0x71,0x7f]
          vfmaddcph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmaddcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x66,0x16,0x97,0x56,0x72,0x80]
          vfmaddcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}

// CHECK: vfmulcph ymm30, ymm29, ymm28
// CHECK: encoding: [0x62,0x06,0x16,0x20,0xd6,0xf4]
          vfmulcph ymm30, ymm29, ymm28

// CHECK: vfmulcph xmm30, xmm29, xmm28
// CHECK: encoding: [0x62,0x06,0x16,0x00,0xd6,0xf4]
          vfmulcph xmm30, xmm29, xmm28

// CHECK: vfmulcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x16,0x27,0xd6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmulcph ymm30 {k7}, ymm29, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmulcph ymm30, ymm29, dword ptr [r9]{1to8}
// CHECK: encoding: [0x62,0x46,0x16,0x30,0xd6,0x31]
          vfmulcph ymm30, ymm29, dword ptr [r9]{1to8}

// CHECK: vfmulcph ymm30, ymm29, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0x66,0x16,0x20,0xd6,0x71,0x7f]
          vfmulcph ymm30, ymm29, ymmword ptr [rcx + 4064]

// CHECK: vfmulcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0x66,0x16,0xb7,0xd6,0x72,0x80]
          vfmulcph ymm30 {k7} {z}, ymm29, dword ptr [rdx - 512]{1to8}

// CHECK: vfmulcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x26,0x16,0x07,0xd6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmulcph xmm30 {k7}, xmm29, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vfmulcph xmm30, xmm29, dword ptr [r9]{1to4}
// CHECK: encoding: [0x62,0x46,0x16,0x10,0xd6,0x31]
          vfmulcph xmm30, xmm29, dword ptr [r9]{1to4}

// CHECK: vfmulcph xmm30, xmm29, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0x66,0x16,0x00,0xd6,0x71,0x7f]
          vfmulcph xmm30, xmm29, xmmword ptr [rcx + 2032]

// CHECK: vfmulcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0x66,0x16,0x97,0xd6,0x72,0x80]
          vfmulcph xmm30 {k7} {z}, xmm29, dword ptr [rdx - 512]{1to4}
