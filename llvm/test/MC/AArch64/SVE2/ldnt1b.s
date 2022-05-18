// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldnt1b z0.s, p0/z, [z1.s]
// CHECK-INST: ldnt1b { z0.s }, p0/z, [z1.s]
// CHECK-ENCODING: [0x20,0xa0,0x1f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 a0 1f 84 <unknown>

ldnt1b z31.s, p7/z, [z31.s, xzr]
// CHECK-INST: ldnt1b { z31.s }, p7/z, [z31.s]
// CHECK-ENCODING: [0xff,0xbf,0x1f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff bf 1f 84 <unknown>

ldnt1b z31.s, p7/z, [z31.s, x0]
// CHECK-INST: ldnt1b { z31.s }, p7/z, [z31.s, x0]
// CHECK-ENCODING: [0xff,0xbf,0x00,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff bf 00 84 <unknown>

ldnt1b z0.d, p0/z, [z1.d]
// CHECK-INST: ldnt1b { z0.d }, p0/z, [z1.d]
// CHECK-ENCODING: [0x20,0xc0,0x1f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 c0 1f c4 <unknown>

ldnt1b z31.d, p7/z, [z31.d, xzr]
// CHECK-INST: ldnt1b { z31.d }, p7/z, [z31.d]
// CHECK-ENCODING: [0xff,0xdf,0x1f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 1f c4 <unknown>

ldnt1b z31.d, p7/z, [z31.d, x0]
// CHECK-INST: ldnt1b { z31.d }, p7/z, [z31.d, x0]
// CHECK-ENCODING: [0xff,0xdf,0x00,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 00 c4 <unknown>

ldnt1b { z0.s }, p0/z, [z1.s]
// CHECK-INST: ldnt1b { z0.s }, p0/z, [z1.s]
// CHECK-ENCODING: [0x20,0xa0,0x1f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 a0 1f 84 <unknown>

ldnt1b { z31.s }, p7/z, [z31.s, xzr]
// CHECK-INST: ldnt1b { z31.s }, p7/z, [z31.s]
// CHECK-ENCODING: [0xff,0xbf,0x1f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff bf 1f 84 <unknown>

ldnt1b { z31.s }, p7/z, [z31.s, x0]
// CHECK-INST: ldnt1b { z31.s }, p7/z, [z31.s, x0]
// CHECK-ENCODING: [0xff,0xbf,0x00,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff bf 00 84 <unknown>

ldnt1b { z0.d }, p0/z, [z1.d]
// CHECK-INST: ldnt1b { z0.d }, p0/z, [z1.d]
// CHECK-ENCODING: [0x20,0xc0,0x1f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 c0 1f c4 <unknown>

ldnt1b { z31.d }, p7/z, [z31.d, xzr]
// CHECK-INST: ldnt1b { z31.d }, p7/z, [z31.d]
// CHECK-ENCODING: [0xff,0xdf,0x1f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 1f c4 <unknown>

ldnt1b { z31.d }, p7/z, [z31.d, x0]
// CHECK-INST: ldnt1b { z31.d }, p7/z, [z31.d, x0]
// CHECK-ENCODING: [0xff,0xdf,0x00,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 00 c4 <unknown>
