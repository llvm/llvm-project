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

ldff1sw { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1sw { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a49f7fff <unknown>

ldff1sw { z31.d }, p7/z, [sp, xzr, lsl #2]
// CHECK-INST: ldff1sw { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a49f7fff <unknown>

ldff1sw { z0.d }, p0/z, [x0, x0, lsl #2]
// CHECK-INST: ldff1sw { z0.d }, p0/z, [x0, x0, lsl #2]
// CHECK-ENCODING: [0x00,0x60,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a4806000 <unknown>

ldff1sw { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ldff1sw { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xbf,0x5f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c55fbfff <unknown>

ldff1sw { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-INST: ldff1sw { z23.d }, p3/z, [x13, z8.d, lsl #2]
// CHECK-ENCODING: [0xb7,0xad,0x68,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c568adb7 <unknown>

ldff1sw { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ldff1sw { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x35,0x15,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5153555 <unknown>

ldff1sw { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ldff1sw { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x35,0x55,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5553555 <unknown>

ldff1sw { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-INST: ldff1sw { z0.d }, p0/z, [x0, z0.d, uxtw #2]
// CHECK-ENCODING: [0x00,0x20,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5202000 <unknown>

ldff1sw { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-INST: ldff1sw { z0.d }, p0/z, [x0, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0x20,0x60,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c5602000 <unknown>

ldff1sw { z31.d }, p7/z, [z31.d, #124]
// CHECK-INST: ldff1sw { z31.d }, p7/z, [z31.d, #124]
// CHECK-ENCODING: [0xff,0xbf,0x3f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c53fbfff <unknown>

ldff1sw { z0.d }, p0/z, [z0.d]
// CHECK-INST: ldff1sw { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c520a000 <unknown>
