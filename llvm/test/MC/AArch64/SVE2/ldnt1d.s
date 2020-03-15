// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldnt1d z0.d, p0/z, [z1.d]
// CHECK-INST: ldnt1d { z0.d }, p0/z, [z1.d]
// CHECK-ENCODING: [0x20,0xc0,0x9f,0xc5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 c0 9f c5 <unknown>

ldnt1d z31.d, p7/z, [z31.d, xzr]
// CHECK-INST: ldnt1d { z31.d }, p7/z, [z31.d]
// CHECK-ENCODING: [0xff,0xdf,0x9f,0xc5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 9f c5 <unknown>

ldnt1d z31.d, p7/z, [z31.d, x0]
// CHECK-INST: ldnt1d { z31.d }, p7/z, [z31.d, x0]
// CHECK-ENCODING: [0xff,0xdf,0x80,0xc5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 80 c5 <unknown>

ldnt1d { z0.d }, p0/z, [z1.d]
// CHECK-INST: ldnt1d { z0.d }, p0/z, [z1.d]
// CHECK-ENCODING: [0x20,0xc0,0x9f,0xc5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 c0 9f c5 <unknown>

ldnt1d { z31.d }, p7/z, [z31.d, xzr]
// CHECK-INST: ldnt1d { z31.d }, p7/z, [z31.d]
// CHECK-ENCODING: [0xff,0xdf,0x9f,0xc5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 9f c5 <unknown>

ldnt1d { z31.d }, p7/z, [z31.d, x0]
// CHECK-INST: ldnt1d { z31.d }, p7/z, [z31.d, x0]
// CHECK-ENCODING: [0xff,0xdf,0x80,0xc5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff df 80 c5 <unknown>
