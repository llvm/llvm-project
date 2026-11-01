// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s --check-prefixes=CHECK,ENCODING
// Check that the printed Intel assembly can be reassembled with the same encoding.
// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding | FileCheck %s --check-prefixes=CHECK,ENCODING

// Cover every gather/scatter prefetch mnemonic with unsized and sized memory,
// high VSIB registers, k1-k7, scaled indices, and compressed displacement.
// Expected bytes are taken from avx512pf-64-att.s.

// CHECK: vgatherpf0dps dword ptr [r8 + zmm20] {k1}
// ENCODING-SAME: encoding: [0x62,0xd2,0x7d,0x41,0xc6,0x0c,0x20]
vgatherpf0dps [r8 + zmm20] {k1}

// CHECK: vgatherpf1dps dword ptr [r9 + zmm19] {k2}
// ENCODING-SAME: encoding: [0x62,0xd2,0x7d,0x42,0xc6,0x14,0x19]
vgatherpf1dps [r9 + zmm19] {k2}

// CHECK: vgatherpf0dpd qword ptr [r10 + ymm18] {k3}
// ENCODING-SAME: encoding: [0x62,0xd2,0xfd,0x43,0xc6,0x0c,0x12]
vgatherpf0dpd [r10 + ymm18] {k3}

// CHECK: vgatherpf1dpd qword ptr [r11 + ymm17] {k4}
// ENCODING-SAME: encoding: [0x62,0xd2,0xfd,0x44,0xc6,0x14,0x0b]
vgatherpf1dpd [r11 + ymm17] {k4}

// CHECK: vgatherpf0qps dword ptr [r12 + zmm16] {k5}
// ENCODING-SAME: encoding: [0x62,0xd2,0x7d,0x45,0xc7,0x0c,0x04]
vgatherpf0qps [r12 + zmm16] {k5}

// CHECK: vgatherpf1qps dword ptr [r13 + zmm15] {k6}
// ENCODING-SAME: encoding: [0x62,0x92,0x7d,0x4e,0xc7,0x54,0x3d,0x00]
vgatherpf1qps [r13 + zmm15] {k6}

// CHECK: vgatherpf0qpd qword ptr [r14 + zmm14] {k7}
// ENCODING-SAME: encoding: [0x62,0x92,0xfd,0x4f,0xc7,0x0c,0x36]
vgatherpf0qpd [r14 + zmm14] {k7}

// CHECK: vgatherpf1qpd qword ptr [r15 + zmm13] {k1}
// ENCODING-SAME: encoding: [0x62,0x92,0xfd,0x49,0xc7,0x14,0x2f]
vgatherpf1qpd [r15 + zmm13] {k1}

// CHECK: vscatterpf0dps dword ptr [r8 + zmm20] {k1}
// ENCODING-SAME: encoding: [0x62,0xd2,0x7d,0x41,0xc6,0x2c,0x20]
vscatterpf0dps [r8 + zmm20] {k1}

// CHECK: vscatterpf1dps dword ptr [r9 + zmm19] {k2}
// ENCODING-SAME: encoding: [0x62,0xd2,0x7d,0x42,0xc6,0x34,0x19]
vscatterpf1dps [r9 + zmm19] {k2}

// CHECK: vscatterpf0dpd qword ptr [r10 + ymm18] {k3}
// ENCODING-SAME: encoding: [0x62,0xd2,0xfd,0x43,0xc6,0x2c,0x12]
vscatterpf0dpd [r10 + ymm18] {k3}

// CHECK: vscatterpf1dpd qword ptr [r11 + ymm17] {k4}
// ENCODING-SAME: encoding: [0x62,0xd2,0xfd,0x44,0xc6,0x34,0x0b]
vscatterpf1dpd [r11 + ymm17] {k4}

// CHECK: vscatterpf0qps dword ptr [r12 + zmm16] {k5}
// ENCODING-SAME: encoding: [0x62,0xd2,0x7d,0x45,0xc7,0x2c,0x04]
vscatterpf0qps [r12 + zmm16] {k5}

// CHECK: vscatterpf1qps dword ptr [r13 + zmm15] {k6}
// ENCODING-SAME: encoding: [0x62,0x92,0x7d,0x4e,0xc7,0x74,0x3d,0x00]
vscatterpf1qps [r13 + zmm15] {k6}

// CHECK: vscatterpf0qpd qword ptr [r14 + zmm14] {k7}
// ENCODING-SAME: encoding: [0x62,0x92,0xfd,0x4f,0xc7,0x2c,0x36]
vscatterpf0qpd [r14 + zmm14] {k7}

// CHECK: vscatterpf1qpd qword ptr [r15 + zmm13] {k1}
// ENCODING-SAME: encoding: [0x62,0x92,0xfd,0x49,0xc7,0x34,0x2f]
vscatterpf1qpd [r15 + zmm13] {k1}

// CHECK: vgatherpf0dpd qword ptr [rcx + 2*ymm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc6,0x4c,0x51,0x01]
vgatherpf0dpd qword ptr [rcx + 2*ymm2 + 8] {k1}

// CHECK: vgatherpf0dps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc6,0x4c,0x51,0x01]
vgatherpf0dps dword ptr [rcx + 2*zmm2 + 4] {k1}

// CHECK: vgatherpf0qpd qword ptr [rcx + 2*zmm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc7,0x4c,0x51,0x01]
vgatherpf0qpd qword ptr [rcx + 2*zmm2 + 8] {k1}

// CHECK: vgatherpf0qps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc7,0x4c,0x51,0x01]
vgatherpf0qps dword ptr [rcx + 2*zmm2 + 4] {k1}

// CHECK: vgatherpf1dpd qword ptr [rcx + 2*ymm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc6,0x54,0x51,0x01]
vgatherpf1dpd qword ptr [rcx + 2*ymm2 + 8] {k1}

// CHECK: vgatherpf1dps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc6,0x54,0x51,0x01]
vgatherpf1dps dword ptr [rcx + 2*zmm2 + 4] {k1}

// CHECK: vgatherpf1qpd qword ptr [rcx + 2*zmm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc7,0x54,0x51,0x01]
vgatherpf1qpd qword ptr [rcx + 2*zmm2 + 8] {k1}

// CHECK: vgatherpf1qps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc7,0x54,0x51,0x01]
vgatherpf1qps dword ptr [rcx + 2*zmm2 + 4] {k1}

// CHECK: vscatterpf0dpd qword ptr [rcx + 2*ymm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc6,0x6c,0x51,0x01]
vscatterpf0dpd qword ptr [rcx + 2*ymm2 + 8] {k1}

// CHECK: vscatterpf0dps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc6,0x6c,0x51,0x01]
vscatterpf0dps dword ptr [rcx + 2*zmm2 + 4] {k1}

// CHECK: vscatterpf0qpd qword ptr [rcx + 2*zmm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc7,0x6c,0x51,0x01]
vscatterpf0qpd qword ptr [rcx + 2*zmm2 + 8] {k1}

// CHECK: vscatterpf0qps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc7,0x6c,0x51,0x01]
vscatterpf0qps dword ptr [rcx + 2*zmm2 + 4] {k1}

// CHECK: vscatterpf1dpd qword ptr [rcx + 2*ymm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc6,0x74,0x51,0x01]
vscatterpf1dpd qword ptr [rcx + 2*ymm2 + 8] {k1}

// CHECK: vscatterpf1dps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc6,0x74,0x51,0x01]
vscatterpf1dps dword ptr [rcx + 2*zmm2 + 4] {k1}

// CHECK: vscatterpf1qpd qword ptr [rcx + 2*zmm2 + 8] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0xfd,0x49,0xc7,0x74,0x51,0x01]
vscatterpf1qpd qword ptr [rcx + 2*zmm2 + 8] {k1}

// CHECK: vscatterpf1qps dword ptr [rcx + 2*zmm2 + 4] {k1}
// ENCODING-SAME: encoding: [0x62,0xf2,0x7d,0x49,0xc7,0x74,0x51,0x01]
vscatterpf1qps dword ptr [rcx + 2*zmm2 + 4] {k1}
