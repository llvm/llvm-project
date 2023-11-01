// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ldnt1sh z0.s, p0/z, [z1.s]
// CHECK-INST: ldnt1sh { z0.s }, p0/z, [z1.s]
// CHECK-ENCODING: [0x20,0x80,0x9f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 849f8020 <unknown>

ldnt1sh z31.s, p7/z, [z31.s, xzr]
// CHECK-INST: ldnt1sh { z31.s }, p7/z, [z31.s]
// CHECK-ENCODING: [0xff,0x9f,0x9f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 849f9fff <unknown>

ldnt1sh z31.s, p7/z, [z31.s, x0]
// CHECK-INST: ldnt1sh { z31.s }, p7/z, [z31.s, x0]
// CHECK-ENCODING: [0xff,0x9f,0x80,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 84809fff <unknown>

ldnt1sh z0.d, p0/z, [z1.d]
// CHECK-INST: ldnt1sh { z0.d }, p0/z, [z1.d]
// CHECK-ENCODING: [0x20,0x80,0x9f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: c49f8020 <unknown>

ldnt1sh z31.d, p7/z, [z31.d, xzr]
// CHECK-INST: ldnt1sh { z31.d }, p7/z, [z31.d]
// CHECK-ENCODING: [0xff,0x9f,0x9f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: c49f9fff <unknown>

ldnt1sh z31.d, p7/z, [z31.d, x0]
// CHECK-INST: ldnt1sh { z31.d }, p7/z, [z31.d, x0]
// CHECK-ENCODING: [0xff,0x9f,0x80,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: c4809fff <unknown>

ldnt1sh { z0.s }, p0/z, [z1.s]
// CHECK-INST: ldnt1sh { z0.s }, p0/z, [z1.s]
// CHECK-ENCODING: [0x20,0x80,0x9f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 849f8020 <unknown>

ldnt1sh { z31.s }, p7/z, [z31.s, xzr]
// CHECK-INST: ldnt1sh { z31.s }, p7/z, [z31.s]
// CHECK-ENCODING: [0xff,0x9f,0x9f,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 849f9fff <unknown>

ldnt1sh { z31.s }, p7/z, [z31.s, x0]
// CHECK-INST: ldnt1sh { z31.s }, p7/z, [z31.s, x0]
// CHECK-ENCODING: [0xff,0x9f,0x80,0x84]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 84809fff <unknown>

ldnt1sh { z0.d }, p0/z, [z1.d]
// CHECK-INST: ldnt1sh { z0.d }, p0/z, [z1.d]
// CHECK-ENCODING: [0x20,0x80,0x9f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: c49f8020 <unknown>

ldnt1sh { z31.d }, p7/z, [z31.d, xzr]
// CHECK-INST: ldnt1sh { z31.d }, p7/z, [z31.d]
// CHECK-ENCODING: [0xff,0x9f,0x9f,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: c49f9fff <unknown>

ldnt1sh { z31.d }, p7/z, [z31.d, x0]
// CHECK-INST: ldnt1sh { z31.d }, p7/z, [z31.d, x0]
// CHECK-ENCODING: [0xff,0x9f,0x80,0xc4]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: c4809fff <unknown>
