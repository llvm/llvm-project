// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

stnt1d z0.d, p0, [z1.d]
// CHECK-INST: stnt1d { z0.d }, p0, [z1.d]
// CHECK-ENCODING: [0x20,0x20,0x9f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 20 9f e5 <unknown>

stnt1d z31.d, p7, [z31.d, xzr]
// CHECK-INST: stnt1d { z31.d }, p7, [z31.d]
// CHECK-ENCODING: [0xff,0x3f,0x9f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 9f e5 <unknown>

stnt1d z31.d, p7, [z31.d, x0]
// CHECK-INST: stnt1d { z31.d }, p7, [z31.d, x0]
// CHECK-ENCODING: [0xff,0x3f,0x80,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 80 e5 <unknown>

stnt1d { z0.d }, p0, [z1.d]
// CHECK-INST: stnt1d { z0.d }, p0, [z1.d]
// CHECK-ENCODING: [0x20,0x20,0x9f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 20 9f e5 <unknown>

stnt1d { z31.d }, p7, [z31.d, xzr]
// CHECK-INST: stnt1d { z31.d }, p7, [z31.d]
// CHECK-ENCODING: [0xff,0x3f,0x9f,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 9f e5 <unknown>

stnt1d { z31.d }, p7, [z31.d, x0]
// CHECK-INST: stnt1d { z31.d }, p7, [z31.d, x0]
// CHECK-ENCODING: [0xff,0x3f,0x80,0xe5]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: ff 3f 80 e5 <unknown>
