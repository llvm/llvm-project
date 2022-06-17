// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ushllb     z0.h, z0.b, #0
// CHECK-INST: ushllb	z0.h, z0.b, #0
// CHECK-ENCODING: [0x00,0xa8,0x08,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 a8 08 45 <unknown>

ushllb     z31.h, z31.b, #7
// CHECK-INST: ushllb	z31.h, z31.b, #7
// CHECK-ENCODING: [0xff,0xab,0x0f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff ab 0f 45 <unknown>

ushllb     z0.s, z0.h, #0
// CHECK-INST: ushllb	z0.s, z0.h, #0
// CHECK-ENCODING: [0x00,0xa8,0x10,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 a8 10 45 <unknown>

ushllb     z31.s, z31.h, #15
// CHECK-INST: ushllb	z31.s, z31.h, #15
// CHECK-ENCODING: [0xff,0xab,0x1f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff ab 1f 45 <unknown>

ushllb     z0.d, z0.s, #0
// CHECK-INST: ushllb	z0.d, z0.s, #0
// CHECK-ENCODING: [0x00,0xa8,0x40,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 00 a8 40 45 <unknown>

ushllb     z31.d, z31.s, #31
// CHECK-INST: ushllb	z31.d, z31.s, #31
// CHECK-ENCODING: [0xff,0xab,0x5f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: ff ab 5f 45 <unknown>
