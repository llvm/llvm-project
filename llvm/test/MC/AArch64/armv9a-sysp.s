// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+d128,+tlb-rmi,+xs < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+tlb-rmi,+xs < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+d128,+tlb-rmi,+xs < %s \
// RUN:        | llvm-objdump -d --mattr=+d128,+tlb-rmi,+xs --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+d128,+tlb-rmi,+xs < %s \
// RUN:   | llvm-objdump -d --mattr=-d128,+tlb-rmi,+xs --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+d128,+tlb-rmi,+xs < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+d128,+tlb-rmi,+xs -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// +tbl-rmi required for RIPA*/RVA*
// +xs required for *NXS

// sysp #<op1>, <Cn>, <Cm>, #<op2>{, <Xt1>, <Xt2>}
// registers with 128-bit formats (op0, op1, Cn, Cm, op2)
// For sysp, op0 is 0

sysp #0, c2, c0, #0, x0, x1// TTBR0_EL1     3  0  2  0  0
// CHECK-INST: sysp #0, c2, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482000      <unknown>

sysp #0, c2, c0, #1, x0, x1// TTBR1_EL1     3  0  2  0  1
// CHECK-INST: sysp #0, c2, c0, #1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482020      <unknown>

sysp #0, c7, c4, #0, x0, x1// PAR_EL1       3  0  7  4  0
// CHECK-INST: sysp #0, c7, c4, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x74,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5487400      <unknown>

sysp #0, c13, c0, #3, x0, x1         // RCWSMASK_EL1  3  0 13  0  3
// CHECK-INST: sysp #0, c13, c0, #3, x0, x1
// CHECK-ENCODING: encoding: [0x60,0xd0,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548d060      <unknown>

sysp #0, c13, c0, #6, x0, x1         // RCWMASK_EL1   3  0 13  0  6
// CHECK-INST: sysp #0, c13, c0, #6, x0, x1
// CHECK-ENCODING: encoding: [0xc0,0xd0,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548d0c0      <unknown>

sysp #4, c2, c0, #0, x0, x1// TTBR0_EL2     3  4  2  0  0
// CHECK-INST: sysp #4, c2, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c2000      <unknown>

sysp #4, c2, c0, #1, x0, x1// TTBR1_EL2     3  4  2  0  1
// CHECK-INST: sysp #4, c2, c0, #1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c2020      <unknown>

sysp #4, c2, c1, #0, x0, x1// VTTBR_EL2     3  4  2  1  0
// CHECK-INST: sysp #4, c2, c1, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x21,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c2100      <unknown>



sysp #0, c2, c0, #0, x0, x1
// CHECK-INST: sysp #0, c2, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482000      <unknown>

sysp #0, c2, c0, #1, x0, x1
// CHECK-INST: sysp #0, c2, c0, #1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482020      <unknown>

sysp #0, c7, c4, #0, x0, x1
// CHECK-INST: sysp #0, c7, c4, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x74,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5487400      <unknown>

sysp #0, c13, c0, #3, x0, x1
// CHECK-INST: sysp #0, c13, c0, #3, x0, x1
// CHECK-ENCODING: encoding: [0x60,0xd0,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548d060      <unknown>

sysp #0, c13, c0, #6, x0, x1
// CHECK-INST: sysp #0, c13, c0, #6, x0, x1
// CHECK-ENCODING: encoding: [0xc0,0xd0,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548d0c0      <unknown>

sysp #4, c2, c0, #0, x0, x1
// CHECK-INST: sysp #4, c2, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c2000      <unknown>

sysp #4, c2, c0, #1, x0, x1
// CHECK-INST: sysp #4, c2, c0, #1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c2020      <unknown>

sysp #4, c2, c1, #0, x0, x1
// CHECK-INST: sysp #4, c2, c1, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x21,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c2100      <unknown>

sysp #0, c2, c0, #0, x0, x1
// CHECK-INST: sysp #0, c2, c0, #0, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482000      <unknown>

sysp #0, c2, c0, #0, x2, x3
// CHECK-INST: sysp #0, c2, c0, #0, x2, x3
// CHECK-ENCODING: encoding: [0x02,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482002      <unknown>

sysp #0, c2, c0, #0, x4, x5
// CHECK-INST: sysp #0, c2, c0, #0, x4, x5
// CHECK-ENCODING: encoding: [0x04,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482004      <unknown>

sysp #0, c2, c0, #0, x6, x7
// CHECK-INST: sysp #0, c2, c0, #0, x6, x7
// CHECK-ENCODING: encoding: [0x06,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482006      <unknown>

sysp #0, c2, c0, #0, x8, x9
// CHECK-INST: sysp #0, c2, c0, #0, x8, x9
// CHECK-ENCODING: encoding: [0x08,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482008      <unknown>

sysp #0, c2, c0, #0, x10, x11
// CHECK-INST: sysp #0, c2, c0, #0, x10, x11
// CHECK-ENCODING: encoding: [0x0a,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548200a      <unknown>

sysp #0, c2, c0, #0, x12, x13
// CHECK-INST: sysp #0, c2, c0, #0, x12, x13
// CHECK-ENCODING: encoding: [0x0c,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548200c      <unknown>

sysp #0, c2, c0, #0, x14, x15
// CHECK-INST: sysp #0, c2, c0, #0, x14, x15
// CHECK-ENCODING: encoding: [0x0e,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548200e      <unknown>

sysp #0, c2, c0, #0, x16, x17
// CHECK-INST: sysp #0, c2, c0, #0, x16, x17
// CHECK-ENCODING: encoding: [0x10,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482010      <unknown>

sysp #0, c2, c0, #0, x18, x19
// CHECK-INST: sysp #0, c2, c0, #0, x18, x19
// CHECK-ENCODING: encoding: [0x12,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482012      <unknown>

sysp #0, c2, c0, #0, x20, x21
// CHECK-INST: sysp #0, c2, c0, #0, x20, x21
// CHECK-ENCODING: encoding: [0x14,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482014      <unknown>

sysp #0, c2, c0, #0, x22, x23
// CHECK-INST: sysp #0, c2, c0, #0, x22, x23
// CHECK-ENCODING: encoding: [0x16,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482016      <unknown>

sysp #0, c2, c0, #0, x24, x25
// CHECK-INST: sysp #0, c2, c0, #0, x24, x25
// CHECK-ENCODING: encoding: [0x18,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5482018      <unknown>

sysp #0, c2, c0, #0, x26, x27
// CHECK-INST: sysp #0, c2, c0, #0, x26, x27
// CHECK-ENCODING: encoding: [0x1a,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201a      <unknown>

sysp #0, c2, c0, #0, x28, x29
// CHECK-INST: sysp #0, c2, c0, #0, x28, x29
// CHECK-ENCODING: encoding: [0x1c,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201c      <unknown>

sysp #0, c2, c0, #0, x30, x31
// CHECK-INST: sysp #0, c2, c0, #0, x30, xzr
// CHECK-ENCODING: encoding: [0x1e,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201e      <unknown>


sysp #0, c2, c0, #0, x31, x31
// CHECK-INST: sysp #0, c2, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201f      <unknown>

sysp #0, c2, c0, #0, xzr, xzr
// CHECK-INST: sysp #0, c2, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201f      <unknown>

sysp #0, c2, c0, #0, x31, xzr
// CHECK-INST: sysp #0, c2, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201f      <unknown>

sysp #0, c2, c0, #0, xzr, x31
// CHECK-INST: sysp #0, c2, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201f      <unknown>

sysp #0, c2, c0, #0
// CHECK-INST: sysp #0, c2, c0, #0
// CHECK-ENCODING: encoding: [0x1f,0x20,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d548201f      <unknown>

tlbip IPAS2E1, x4, x5
// CHECK-INST: tlbip ipas2e1, x4, x5
// CHECK-ENCODING: encoding: [0x24,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c8424      <unknown>

tlbip IPAS2E1NXS, x4, x5
// CHECK-INST: tlbip ipas2e1nxs, x4, x5
// CHECK-ENCODING: encoding: [0x24,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c9424      <unknown>

tlbip IPAS2E1IS, x4, x5
// CHECK-INST: tlbip ipas2e1is, x4, x5
// CHECK-ENCODING: encoding: [0x24,0x80,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c8024      <unknown>

tlbip IPAS2E1ISNXS, x4, x5
// CHECK-INST: tlbip ipas2e1isnxs, x4, x5
// CHECK-ENCODING: encoding: [0x24,0x90,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c9024      <unknown>

tlbip IPAS2E1OS, x4, x5
// CHECK-INST: tlbip ipas2e1os, x4, x5
// CHECK-ENCODING: encoding: [0x04,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c8404      <unknown>

tlbip IPAS2E1OSNXS, x4, x5
// CHECK-INST: tlbip ipas2e1osnxs, x4, x5
// CHECK-ENCODING: encoding: [0x04,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c9404      <unknown>

tlbip IPAS2LE1, x4, x5
// CHECK-INST: tlbip ipas2le1, x4, x5
// CHECK-ENCODING: encoding: [0xa4,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c84a4      <unknown>

tlbip IPAS2LE1NXS, x4, x5
// CHECK-INST: tlbip ipas2le1nxs, x4, x5
// CHECK-ENCODING: encoding: [0xa4,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c94a4      <unknown>

tlbip IPAS2LE1IS, x4, x5
// CHECK-INST: tlbip ipas2le1is, x4, x5
// CHECK-ENCODING: encoding: [0xa4,0x80,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c80a4      <unknown>

tlbip IPAS2LE1ISNXS, x4, x5
// CHECK-INST: tlbip ipas2le1isnxs, x4, x5
// CHECK-ENCODING: encoding: [0xa4,0x90,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c90a4      <unknown>

tlbip IPAS2LE1OS, x4, x5
// CHECK-INST: tlbip ipas2le1os, x4, x5
// CHECK-ENCODING: encoding: [0x84,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c8484      <unknown>

tlbip IPAS2LE1OSNXS, x4, x5
// CHECK-INST: tlbip ipas2le1osnxs, x4, x5
// CHECK-ENCODING: encoding: [0x84,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c9484      <unknown>

tlbip VAE1, x8, x9
// CHECK-INST: tlbip vae1, x8, x9
// CHECK-ENCODING: encoding: [0x28,0x87,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488728      <unknown>

tlbip VAE1NXS, x8, x9
// CHECK-INST: tlbip vae1nxs, x8, x9
// CHECK-ENCODING: encoding: [0x28,0x97,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489728      <unknown>

tlbip VAE1IS, x8, x9
// CHECK-INST: tlbip vae1is, x8, x9
// CHECK-ENCODING: encoding: [0x28,0x83,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488328      <unknown>

tlbip VAE1ISNXS, x8, x9
// CHECK-INST: tlbip vae1isnxs, x8, x9
// CHECK-ENCODING: encoding: [0x28,0x93,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489328      <unknown>

tlbip VAE1OS, x8, x9
// CHECK-INST: tlbip vae1os, x8, x9
// CHECK-ENCODING: encoding: [0x28,0x81,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488128      <unknown>

tlbip VAE1OSNXS, x8, x9
// CHECK-INST: tlbip vae1osnxs, x8, x9
// CHECK-ENCODING: encoding: [0x28,0x91,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489128      <unknown>

tlbip VALE1, x8, x9
// CHECK-INST: tlbip vale1, x8, x9
// CHECK-ENCODING: encoding: [0xa8,0x87,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54887a8      <unknown>

tlbip VALE1NXS, x8, x9
// CHECK-INST: tlbip vale1nxs, x8, x9
// CHECK-ENCODING: encoding: [0xa8,0x97,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54897a8      <unknown>

tlbip VALE1IS, x8, x9
// CHECK-INST: tlbip vale1is, x8, x9
// CHECK-ENCODING: encoding: [0xa8,0x83,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54883a8      <unknown>

tlbip VALE1ISNXS, x8, x9
// CHECK-INST: tlbip vale1isnxs, x8, x9
// CHECK-ENCODING: encoding: [0xa8,0x93,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54893a8      <unknown>

tlbip VALE1OS, x8, x9
// CHECK-INST: tlbip vale1os, x8, x9
// CHECK-ENCODING: encoding: [0xa8,0x81,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54881a8      <unknown>

tlbip VALE1OSNXS, x8, x9
// CHECK-INST: tlbip vale1osnxs, x8, x9
// CHECK-ENCODING: encoding: [0xa8,0x91,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54891a8      <unknown>

tlbip VAAE1, x8, x9
// CHECK-INST: tlbip vaae1, x8, x9
// CHECK-ENCODING: encoding: [0x68,0x87,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488768      <unknown>

tlbip VAAE1NXS, x8, x9
// CHECK-INST: tlbip vaae1nxs, x8, x9
// CHECK-ENCODING: encoding: [0x68,0x97,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489768      <unknown>

tlbip VAAE1IS, x8, x9
// CHECK-INST: tlbip vaae1is, x8, x9
// CHECK-ENCODING: encoding: [0x68,0x83,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488368      <unknown>

tlbip VAAE1ISNXS, x8, x9
// CHECK-INST: tlbip vaae1isnxs, x8, x9
// CHECK-ENCODING: encoding: [0x68,0x93,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489368      <unknown>

tlbip VAAE1OS, x8, x9
// CHECK-INST: tlbip vaae1os, x8, x9
// CHECK-ENCODING: encoding: [0x68,0x81,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488168      <unknown>

tlbip VAAE1OSNXS, x8, x9
// CHECK-INST: tlbip vaae1osnxs, x8, x9
// CHECK-ENCODING: encoding: [0x68,0x91,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489168      <unknown>

tlbip VAALE1, x8, x9
// CHECK-INST: tlbip vaale1, x8, x9
// CHECK-ENCODING: encoding: [0xe8,0x87,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54887e8      <unknown>

tlbip VAALE1NXS, x8, x9
// CHECK-INST: tlbip vaale1nxs, x8, x9
// CHECK-ENCODING: encoding: [0xe8,0x97,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54897e8      <unknown>

tlbip VAALE1IS, x8, x9
// CHECK-INST: tlbip vaale1is, x8, x9
// CHECK-ENCODING: encoding: [0xe8,0x83,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54883e8      <unknown>

tlbip VAALE1ISNXS, x8, x9
// CHECK-INST: tlbip vaale1isnxs, x8, x9
// CHECK-ENCODING: encoding: [0xe8,0x93,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54893e8      <unknown>

tlbip VAALE1OS, x8, x9
// CHECK-INST: tlbip vaale1os, x8, x9
// CHECK-ENCODING: encoding: [0xe8,0x81,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54881e8      <unknown>

tlbip VAALE1OSNXS, x8, x9
// CHECK-INST: tlbip vaale1osnxs, x8, x9
// CHECK-ENCODING: encoding: [0xe8,0x91,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54891e8      <unknown>

tlbip VAE2, x14, x15
// CHECK-INST: tlbip vae2, x14, x15
// CHECK-ENCODING: encoding: [0x2e,0x87,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c872e      <unknown>

tlbip VAE2NXS, x14, x15
// CHECK-INST: tlbip vae2nxs, x14, x15
// CHECK-ENCODING: encoding: [0x2e,0x97,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c972e      <unknown>

tlbip VAE2IS, x14, x15
// CHECK-INST: tlbip vae2is, x14, x15
// CHECK-ENCODING: encoding: [0x2e,0x83,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c832e      <unknown>

tlbip VAE2ISNXS, x14, x15
// CHECK-INST: tlbip vae2isnxs, x14, x15
// CHECK-ENCODING: encoding: [0x2e,0x93,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c932e      <unknown>

tlbip VAE2OS, x14, x15
// CHECK-INST: tlbip vae2os, x14, x15
// CHECK-ENCODING: encoding: [0x2e,0x81,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c812e      <unknown>

tlbip VAE2OSNXS, x14, x15
// CHECK-INST: tlbip vae2osnxs, x14, x15
// CHECK-ENCODING: encoding: [0x2e,0x91,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c912e      <unknown>

tlbip VALE2, x14, x15
// CHECK-INST: tlbip vale2, x14, x15
// CHECK-ENCODING: encoding: [0xae,0x87,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c87ae      <unknown>

tlbip VALE2NXS, x14, x15
// CHECK-INST: tlbip vale2nxs, x14, x15
// CHECK-ENCODING: encoding: [0xae,0x97,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c97ae      <unknown>

tlbip VALE2IS, x14, x15
// CHECK-INST: tlbip vale2is, x14, x15
// CHECK-ENCODING: encoding: [0xae,0x83,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c83ae      <unknown>

tlbip VALE2ISNXS, x14, x15
// CHECK-INST: tlbip vale2isnxs, x14, x15
// CHECK-ENCODING: encoding: [0xae,0x93,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c93ae      <unknown>

tlbip VALE2OS, x14, x15
// CHECK-INST: tlbip vale2os, x14, x15
// CHECK-ENCODING: encoding: [0xae,0x81,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c81ae      <unknown>

tlbip VALE2OSNXS, x14, x15
// CHECK-INST: tlbip vale2osnxs, x14, x15
// CHECK-ENCODING: encoding: [0xae,0x91,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c91ae      <unknown>

tlbip VAE3, x24, x25
// CHECK-INST: tlbip vae3, x24, x25
// CHECK-ENCODING: encoding: [0x38,0x87,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e8738      <unknown>

tlbip VAE3NXS, x24, x25
// CHECK-INST: tlbip vae3nxs, x24, x25
// CHECK-ENCODING: encoding: [0x38,0x97,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e9738      <unknown>

tlbip VAE3IS, x24, x25
// CHECK-INST: tlbip vae3is, x24, x25
// CHECK-ENCODING: encoding: [0x38,0x83,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e8338      <unknown>

tlbip VAE3ISNXS, x24, x25
// CHECK-INST: tlbip vae3isnxs, x24, x25
// CHECK-ENCODING: encoding: [0x38,0x93,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e9338      <unknown>

tlbip VAE3OS, x24, x25
// CHECK-INST: tlbip vae3os, x24, x25
// CHECK-ENCODING: encoding: [0x38,0x81,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e8138      <unknown>

tlbip VAE3OSNXS, x24, x25
// CHECK-INST: tlbip vae3osnxs, x24, x25
// CHECK-ENCODING: encoding: [0x38,0x91,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e9138      <unknown>

tlbip VALE3, x24, x25
// CHECK-INST: tlbip vale3, x24, x25
// CHECK-ENCODING: encoding: [0xb8,0x87,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e87b8      <unknown>

tlbip VALE3NXS, x24, x25
// CHECK-INST: tlbip vale3nxs, x24, x25
// CHECK-ENCODING: encoding: [0xb8,0x97,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e97b8      <unknown>

tlbip VALE3IS, x24, x25
// CHECK-INST: tlbip vale3is, x24, x25
// CHECK-ENCODING: encoding: [0xb8,0x83,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e83b8      <unknown>

tlbip VALE3ISNXS, x24, x25
// CHECK-INST: tlbip vale3isnxs, x24, x25
// CHECK-ENCODING: encoding: [0xb8,0x93,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e93b8      <unknown>

tlbip VALE3OS, x24, x25
// CHECK-INST: tlbip vale3os, x24, x25
// CHECK-ENCODING: encoding: [0xb8,0x81,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e81b8      <unknown>

tlbip VALE3OSNXS, x24, x25
// CHECK-INST: tlbip vale3osnxs, x24, x25
// CHECK-ENCODING: encoding: [0xb8,0x91,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e91b8      <unknown>

tlbip RVAE1, x18, x19
// CHECK-INST: tlbip rvae1, x18, x19
// CHECK-ENCODING: encoding: [0x32,0x86,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488632      <unknown>

tlbip RVAE1NXS, x18, x19
// CHECK-INST: tlbip rvae1nxs, x18, x19
// CHECK-ENCODING: encoding: [0x32,0x96,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489632      <unknown>

tlbip RVAE1IS, x18, x19
// CHECK-INST: tlbip rvae1is, x18, x19
// CHECK-ENCODING: encoding: [0x32,0x82,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488232      <unknown>

tlbip RVAE1ISNXS, x18, x19
// CHECK-INST: tlbip rvae1isnxs, x18, x19
// CHECK-ENCODING: encoding: [0x32,0x92,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489232      <unknown>

tlbip RVAE1OS, x18, x19
// CHECK-INST: tlbip rvae1os, x18, x19
// CHECK-ENCODING: encoding: [0x32,0x85,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488532      <unknown>

tlbip RVAE1OSNXS, x18, x19
// CHECK-INST: tlbip rvae1osnxs, x18, x19
// CHECK-ENCODING: encoding: [0x32,0x95,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489532      <unknown>

tlbip RVAAE1, x18, x19
// CHECK-INST: tlbip rvaae1, x18, x19
// CHECK-ENCODING: encoding: [0x72,0x86,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488672      <unknown>

tlbip RVAAE1NXS, x18, x19
// CHECK-INST: tlbip rvaae1nxs, x18, x19
// CHECK-ENCODING: encoding: [0x72,0x96,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489672      <unknown>

tlbip RVAAE1IS, x18, x19
// CHECK-INST: tlbip rvaae1is, x18, x19
// CHECK-ENCODING: encoding: [0x72,0x82,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488272      <unknown>

tlbip RVAAE1ISNXS, x18, x19
// CHECK-INST: tlbip rvaae1isnxs, x18, x19
// CHECK-ENCODING: encoding: [0x72,0x92,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489272      <unknown>

tlbip RVAAE1OS, x18, x19
// CHECK-INST: tlbip rvaae1os, x18, x19
// CHECK-ENCODING: encoding: [0x72,0x85,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5488572      <unknown>

tlbip RVAAE1OSNXS, x18, x19
// CHECK-INST: tlbip rvaae1osnxs, x18, x19
// CHECK-ENCODING: encoding: [0x72,0x95,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5489572      <unknown>

tlbip RVALE1, x18, x19
// CHECK-INST: tlbip rvale1, x18, x19
// CHECK-ENCODING: encoding: [0xb2,0x86,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54886b2      <unknown>

tlbip RVALE1NXS, x18, x19
// CHECK-INST: tlbip rvale1nxs, x18, x19
// CHECK-ENCODING: encoding: [0xb2,0x96,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54896b2      <unknown>

tlbip RVALE1IS, x18, x19
// CHECK-INST: tlbip rvale1is, x18, x19
// CHECK-ENCODING: encoding: [0xb2,0x82,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54882b2      <unknown>

tlbip RVALE1ISNXS, x18, x19
// CHECK-INST: tlbip rvale1isnxs, x18, x19
// CHECK-ENCODING: encoding: [0xb2,0x92,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54892b2      <unknown>

tlbip RVALE1OS, x18, x19
// CHECK-INST: tlbip rvale1os, x18, x19
// CHECK-ENCODING: encoding: [0xb2,0x85,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54885b2      <unknown>

tlbip RVALE1OSNXS, x18, x19
// CHECK-INST: tlbip rvale1osnxs, x18, x19
// CHECK-ENCODING: encoding: [0xb2,0x95,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54895b2      <unknown>

tlbip RVAALE1, x18, x19
// CHECK-INST: tlbip rvaale1, x18, x19
// CHECK-ENCODING: encoding: [0xf2,0x86,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54886f2      <unknown>

tlbip RVAALE1NXS, x18, x19
// CHECK-INST: tlbip rvaale1nxs, x18, x19
// CHECK-ENCODING: encoding: [0xf2,0x96,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54896f2      <unknown>

tlbip RVAALE1IS, x18, x19
// CHECK-INST: tlbip rvaale1is, x18, x19
// CHECK-ENCODING: encoding: [0xf2,0x82,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54882f2      <unknown>

tlbip RVAALE1ISNXS, x18, x19
// CHECK-INST: tlbip rvaale1isnxs, x18, x19
// CHECK-ENCODING: encoding: [0xf2,0x92,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54892f2      <unknown>

tlbip RVAALE1OS, x18, x19
// CHECK-INST: tlbip rvaale1os, x18, x19
// CHECK-ENCODING: encoding: [0xf2,0x85,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54885f2      <unknown>

tlbip RVAALE1OSNXS, x18, x19
// CHECK-INST: tlbip rvaale1osnxs, x18, x19
// CHECK-ENCODING: encoding: [0xf2,0x95,0x48,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54895f2      <unknown>

tlbip RVAE2, x28, x29
// CHECK-INST: tlbip rvae2, x28, x29
// CHECK-ENCODING: encoding: [0x3c,0x86,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c863c      <unknown>

tlbip RVAE2NXS, x28, x29
// CHECK-INST: tlbip rvae2nxs, x28, x29
// CHECK-ENCODING: encoding: [0x3c,0x96,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c963c      <unknown>

tlbip RVAE2IS, x28, x29
// CHECK-INST: tlbip rvae2is, x28, x29
// CHECK-ENCODING: encoding: [0x3c,0x82,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c823c      <unknown>

tlbip RVAE2ISNXS, x28, x29
// CHECK-INST: tlbip rvae2isnxs, x28, x29
// CHECK-ENCODING: encoding: [0x3c,0x92,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c923c      <unknown>

tlbip RVAE2OS, x28, x29
// CHECK-INST: tlbip rvae2os, x28, x29
// CHECK-ENCODING: encoding: [0x3c,0x85,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c853c      <unknown>

tlbip RVAE2OSNXS, x28, x29
// CHECK-INST: tlbip rvae2osnxs, x28, x29
// CHECK-ENCODING: encoding: [0x3c,0x95,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c953c      <unknown>

tlbip RVALE2, x28, x29
// CHECK-INST: tlbip rvale2, x28, x29
// CHECK-ENCODING: encoding: [0xbc,0x86,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c86bc      <unknown>

tlbip RVALE2NXS, x28, x29
// CHECK-INST: tlbip rvale2nxs, x28, x29
// CHECK-ENCODING: encoding: [0xbc,0x96,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c96bc      <unknown>

tlbip RVALE2IS, x28, x29
// CHECK-INST: tlbip rvale2is, x28, x29
// CHECK-ENCODING: encoding: [0xbc,0x82,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c82bc      <unknown>

tlbip RVALE2ISNXS, x28, x29
// CHECK-INST: tlbip rvale2isnxs, x28, x29
// CHECK-ENCODING: encoding: [0xbc,0x92,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c92bc      <unknown>

tlbip RVALE2OS, x28, x29
// CHECK-INST: tlbip rvale2os, x28, x29
// CHECK-ENCODING: encoding: [0xbc,0x85,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c85bc      <unknown>

tlbip RVALE2OSNXS, x28, x29
// CHECK-INST: tlbip rvale2osnxs, x28, x29
// CHECK-ENCODING: encoding: [0xbc,0x95,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c95bc      <unknown>

tlbip RVAE3, x10, x11
// CHECK-INST: tlbip rvae3, x10, x11
// CHECK-ENCODING: encoding: [0x2a,0x86,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e862a      <unknown>

tlbip RVAE3NXS, x10, x11
// CHECK-INST: tlbip rvae3nxs, x10, x11
// CHECK-ENCODING: encoding: [0x2a,0x96,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e962a      <unknown>

tlbip RVAE3IS, x10, x11
// CHECK-INST: tlbip rvae3is, x10, x11
// CHECK-ENCODING: encoding: [0x2a,0x82,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e822a      <unknown>

tlbip RVAE3ISNXS, x10, x11
// CHECK-INST: tlbip rvae3isnxs, x10, x11
// CHECK-ENCODING: encoding: [0x2a,0x92,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e922a      <unknown>

tlbip RVAE3OS, x10, x11
// CHECK-INST: tlbip rvae3os, x10, x11
// CHECK-ENCODING: encoding: [0x2a,0x85,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e852a      <unknown>

tlbip RVAE3OSNXS, x10, x11
// CHECK-INST: tlbip rvae3osnxs, x10, x11
// CHECK-ENCODING: encoding: [0x2a,0x95,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e952a      <unknown>

tlbip RVALE3, x10, x11
// CHECK-INST: tlbip rvale3, x10, x11
// CHECK-ENCODING: encoding: [0xaa,0x86,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e86aa      <unknown>

tlbip RVALE3NXS, x10, x11
// CHECK-INST: tlbip rvale3nxs, x10, x11
// CHECK-ENCODING: encoding: [0xaa,0x96,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e96aa      <unknown>

tlbip RVALE3IS, x10, x11
// CHECK-INST: tlbip rvale3is, x10, x11
// CHECK-ENCODING: encoding: [0xaa,0x82,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e82aa      <unknown>

tlbip RVALE3ISNXS, x10, x11
// CHECK-INST: tlbip rvale3isnxs, x10, x11
// CHECK-ENCODING: encoding: [0xaa,0x92,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e92aa      <unknown>

tlbip RVALE3OS, x10, x11
// CHECK-INST: tlbip rvale3os, x10, x11
// CHECK-ENCODING: encoding: [0xaa,0x85,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e85aa      <unknown>

tlbip RVALE3OSNXS, x10, x11
// CHECK-INST: tlbip rvale3osnxs, x10, x11
// CHECK-ENCODING: encoding: [0xaa,0x95,0x4e,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54e95aa      <unknown>

tlbip RIPAS2E1, x20, x21
// CHECK-INST: tlbip ripas2e1, x20, x21
// CHECK-ENCODING: encoding: [0x54,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c8454      <unknown>

tlbip RIPAS2E1NXS, x20, x21
// CHECK-INST: tlbip ripas2e1nxs, x20, x21
// CHECK-ENCODING: encoding: [0x54,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c9454      <unknown>

tlbip RIPAS2E1IS, x20, x21
// CHECK-INST: tlbip ripas2e1is, x20, x21
// CHECK-ENCODING: encoding: [0x54,0x80,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c8054      <unknown>

tlbip RIPAS2E1ISNXS, x20, x21
// CHECK-INST: tlbip ripas2e1isnxs, x20, x21
// CHECK-ENCODING: encoding: [0x54,0x90,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c9054      <unknown>

tlbip RIPAS2E1OS, x20, x21
// CHECK-INST: tlbip ripas2e1os, x20, x21
// CHECK-ENCODING: encoding: [0x74,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c8474      <unknown>

tlbip RIPAS2E1OSNXS, x20, x21
// CHECK-INST: tlbip ripas2e1osnxs, x20, x21
// CHECK-ENCODING: encoding: [0x74,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c9474      <unknown>

tlbip RIPAS2LE1, x20, x21
// CHECK-INST: tlbip ripas2le1, x20, x21
// CHECK-ENCODING: encoding: [0xd4,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c84d4      <unknown>

tlbip RIPAS2LE1NXS, x20, x21
// CHECK-INST: tlbip ripas2le1nxs, x20, x21
// CHECK-ENCODING: encoding: [0xd4,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c94d4      <unknown>

tlbip RIPAS2LE1IS, x20, x21
// CHECK-INST: tlbip ripas2le1is, x20, x21
// CHECK-ENCODING: encoding: [0xd4,0x80,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c80d4      <unknown>

tlbip RIPAS2LE1ISNXS, x20, x21
// CHECK-INST: tlbip ripas2le1isnxs, x20, x21
// CHECK-ENCODING: encoding: [0xd4,0x90,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c90d4      <unknown>

tlbip RIPAS2LE1OS, x20, x21
// CHECK-INST: tlbip ripas2le1os, x20, x21
// CHECK-ENCODING: encoding: [0xf4,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c84f4      <unknown>

tlbip RIPAS2LE1OSNXS, x20, x21
// CHECK-INST: tlbip ripas2le1osnxs, x20, x21
// CHECK-ENCODING: encoding: [0xf4,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c94f4      <unknown>

tlbip RIPAS2LE1OS, xzr, xzr
// CHECK-INST: tlbip ripas2le1os, xzr, xzr
// CHECK-ENCODING: encoding: [0xff,0x84,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c84ff      <unknown>

tlbip RIPAS2LE1OSNXS, xzr, xzr
// CHECK-INST: tlbip ripas2le1osnxs, xzr, xzr
// CHECK-ENCODING: encoding: [0xff,0x94,0x4c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d54c94ff      <unknown>
