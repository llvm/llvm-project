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

xar     z0.b, z0.b, z1.b, #1
// CHECK-INST: xar	z0.b, z0.b, z1.b, #1
// CHECK-ENCODING: [0x20,0x34,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 34 2f 04 <unknown>

xar     z31.b, z31.b, z30.b, #8
// CHECK-INST: xar	z31.b, z31.b, z30.b, #8
// CHECK-ENCODING: [0xdf,0x37,0x28,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 37 28 04 <unknown>

xar     z0.h, z0.h, z1.h, #1
// CHECK-INST: xar	z0.h, z0.h, z1.h, #1
// CHECK-ENCODING: [0x20,0x34,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 34 3f 04 <unknown>

xar     z31.h, z31.h, z30.h, #16
// CHECK-INST: xar	z31.h, z31.h, z30.h, #16
// CHECK-ENCODING: [0xdf,0x37,0x30,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 37 30 04 <unknown>

xar     z0.s, z0.s, z1.s, #1
// CHECK-INST: xar	z0.s, z0.s, z1.s, #1
// CHECK-ENCODING: [0x20,0x34,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 34 7f 04 <unknown>

xar     z31.s, z31.s, z30.s, #32
// CHECK-INST: xar	z31.s, z31.s, z30.s, #32
// CHECK-ENCODING: [0xdf,0x37,0x60,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 37 60 04 <unknown>

xar     z0.d, z0.d, z1.d, #1
// CHECK-INST: xar	z0.d, z0.d, z1.d, #1
// CHECK-ENCODING: [0x20,0x34,0xff,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 34 ff 04 <unknown>

xar     z31.d, z31.d, z30.d, #64
// CHECK-INST: xar	z31.d, z31.d, z30.d, #64
// CHECK-ENCODING: [0xdf,0x37,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 37 a0 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z7
// CHECK-INST: movprfx z31, z7
// CHECK-ENCODING: [0xff,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bc 20 04 <unknown>

xar     z31.d, z31.d, z30.d, #64
// CHECK-INST: xar     z31.d, z31.d, z30.d, #64
// CHECK-ENCODING: [0xdf,0x37,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 37 a0 04 <unknown>
