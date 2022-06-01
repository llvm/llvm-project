// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sli     z0.b, z0.b, #0
// CHECK-INST: sli	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xf4,0x08,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 f4 08 45 <unknown>

sli     z31.b, z31.b, #7
// CHECK-INST: sli	z31.b, z31.b, #7
// CHECK-ENCODING: [0xff,0xf7,0x0f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff f7 0f 45 <unknown>

sli     z0.h, z0.h, #0
// CHECK-INST: sli	z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0xf4,0x10,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 f4 10 45 <unknown>

sli     z31.h, z31.h, #15
// CHECK-INST: sli	z31.h, z31.h, #15
// CHECK-ENCODING: [0xff,0xf7,0x1f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff f7 1f 45 <unknown>

sli     z0.s, z0.s, #0
// CHECK-INST: sli	z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0xf4,0x40,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 f4 40 45 <unknown>

sli     z31.s, z31.s, #31
// CHECK-INST: sli	z31.s, z31.s, #31
// CHECK-ENCODING: [0xff,0xf7,0x5f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff f7 5f 45 <unknown>

sli     z0.d, z0.d, #0
// CHECK-INST: sli	z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0xf4,0x80,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 f4 80 45 <unknown>

sli     z31.d, z31.d, #63
// CHECK-INST: sli	z31.d, z31.d, #63
// CHECK-ENCODING: [0xff,0xf7,0xdf,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff f7 df 45 <unknown>
