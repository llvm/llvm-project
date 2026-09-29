// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldff1w  { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1w  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x7f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a57f7fff <unknown>

ldff1w  { z31.s }, p7/z, [sp]
// CHECK-INST: ldff1w  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x5f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a55f7fff <unknown>

ldff1w  { z31.d }, p7/z, [sp, xzr, lsl #2]
// CHECK-INST: ldff1w  { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x7f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a57f7fff <unknown>

ldff1w  { z31.s }, p7/z, [sp, xzr, lsl #2]
// CHECK-INST: ldff1w  { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x5f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a55f7fff <unknown>

ldff1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ldff1w  { z0.s }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x60,0x40,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5406000 <unknown>

ldff1w  { z0.d }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ldff1w  { z0.d }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x60,0x60,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5606000 <unknown>

ldff1w  { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ldff1w  { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x60,0x00,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 85006000 <unknown>

ldff1w  { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ldff1w  { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x60,0x40,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 85406000 <unknown>

ldff1w  { z31.s }, p7/z, [sp, z31.s, uxtw #2]
// CHECK-INST: ldff1w  { z31.s }, p7/z, [sp, z31.s, uxtw #2]
// CHECK-ENCODING: [0xff,0x7f,0x3f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 853f7fff <unknown>

ldff1w  { z31.s }, p7/z, [sp, z31.s, sxtw #2]
// CHECK-INST: ldff1w  { z31.s }, p7/z, [sp, z31.s, sxtw #2]
// CHECK-ENCODING: [0xff,0x7f,0x7f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 857f7fff <unknown>

ldff1w  { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ldff1w  { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xff,0x5f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c55fffff <unknown>

ldff1w  { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-INST: ldff1w  { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-ENCODING: [0xb7,0xed,0x68,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c568edb7 <unknown>

ldff1w  { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ldff1w  { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x75,0x15,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5157555 <unknown>

ldff1w  { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ldff1w  { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x75,0x55,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5557555 <unknown>

ldff1w  { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-INST: ldff1w  { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-ENCODING: [0x00,0x60,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5206000 <unknown>

ldff1w  { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-INST: ldff1w  { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0x60,0x60,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5606000 <unknown>

ldff1w  { z31.s }, p7/z, [z31.s, #124]
// CHECK-INST: ldff1w  { z31.s }, p7/z, [z31.s, #124]
// CHECK-ENCODING: [0xff,0xff,0x3f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 853fffff <unknown>

ldff1w  { z0.s }, p0/z, [z0.s]
// CHECK-INST: ldff1w  { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xe0,0x20,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 8520e000 <unknown>

ldff1w  { z31.d }, p7/z, [z31.d, #124]
// CHECK-INST: ldff1w  { z31.d }, p7/z, [z31.d, #124]
// CHECK-ENCODING: [0xff,0xff,0x3f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c53fffff <unknown>

ldff1w  { z0.d }, p0/z, [z0.d]
// CHECK-INST: ldff1w  { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xe0,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c520e000 <unknown>
