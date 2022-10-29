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

ldff1sb { z31.h }, p7/z, [sp]
// CHECK-INST: ldff1sb { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xdf,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5df7fff <unknown>

ldff1sb { z31.s }, p7/z, [sp]
// CHECK-INST: ldff1sb { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xbf,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5bf7fff <unknown>

ldff1sb { z31.d }, p7/z, [sp]
// CHECK-INST: ldff1sb { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a59f7fff <unknown>

ldff1sb { z31.h }, p7/z, [sp, xzr]
// CHECK-INST: ldff1sb { z31.h }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xdf,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5df7fff <unknown>

ldff1sb { z31.s }, p7/z, [sp, xzr]
// CHECK-INST: ldff1sb { z31.s }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0xbf,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5bf7fff <unknown>

ldff1sb { z31.d }, p7/z, [sp, xzr]
// CHECK-INST: ldff1sb { z31.d }, p7/z, [sp]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a59f7fff <unknown>

ldff1sb { z0.h }, p0/z, [x0, x0]
// CHECK-INST: ldff1sb { z0.h }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0xc0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5c06000 <unknown>

ldff1sb { z0.s }, p0/z, [x0, x0]
// CHECK-INST: ldff1sb { z0.s }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0xa0,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5a06000 <unknown>

ldff1sb { z0.d }, p0/z, [x0, x0]
// CHECK-INST: ldff1sb { z0.d }, p0/z, [x0, x0]
// CHECK-ENCODING: [0x00,0x60,0x80,0xa5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a5806000 <unknown>

ldff1sb   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-INST: ldff1sb   { z0.s }, p0/z, [x0, z0.s, uxtw]
// CHECK-ENCODING: [0x00,0x20,0x00,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84002000 <unknown>

ldff1sb   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-INST: ldff1sb   { z0.s }, p0/z, [x0, z0.s, sxtw]
// CHECK-ENCODING: [0x00,0x20,0x40,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 84402000 <unknown>

ldff1sb { z31.d }, p7/z, [sp, z31.d]
// CHECK-INST: ldff1sb { z31.d }, p7/z, [sp, z31.d]
// CHECK-ENCODING: [0xff,0xbf,0x5f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c45fbfff <unknown>

ldff1sb { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-INST: ldff1sb { z21.d }, p5/z, [x10, z21.d, uxtw]
// CHECK-ENCODING: [0x55,0x35,0x15,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4153555 <unknown>

ldff1sb { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-INST: ldff1sb { z21.d }, p5/z, [x10, z21.d, sxtw]
// CHECK-ENCODING: [0x55,0x35,0x55,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c4553555 <unknown>

ldff1sb { z31.s }, p7/z, [z31.s, #31]
// CHECK-INST: ldff1sb { z31.s }, p7/z, [z31.s, #31]
// CHECK-ENCODING: [0xff,0xbf,0x3f,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 843fbfff <unknown>

ldff1sb { z0.s }, p0/z, [z0.s]
// CHECK-INST: ldff1sb { z0.s }, p0/z, [z0.s]
// CHECK-ENCODING: [0x00,0xa0,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 8420a000 <unknown>

ldff1sb { z31.d }, p7/z, [z31.d, #31]
// CHECK-INST: ldff1sb { z31.d }, p7/z, [z31.d, #31]
// CHECK-ENCODING: [0xff,0xbf,0x3f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c43fbfff <unknown>

ldff1sb { z0.d }, p0/z, [z0.d]
// CHECK-INST: ldff1sb { z0.d }, p0/z, [z0.d]
// CHECK-ENCODING: [0x00,0xa0,0x20,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: c420a000 <unknown>
