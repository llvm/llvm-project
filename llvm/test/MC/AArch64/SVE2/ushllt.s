// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ushllt     z0.h, z0.b, #0
// CHECK-INST: ushllt	z0.h, z0.b, #0
// CHECK-ENCODING: [0x00,0xac,0x08,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4508ac00 <unknown>

ushllt     z31.h, z31.b, #7
// CHECK-INST: ushllt	z31.h, z31.b, #7
// CHECK-ENCODING: [0xff,0xaf,0x0f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 450fafff <unknown>

ushllt     z0.s, z0.h, #0
// CHECK-INST: ushllt	z0.s, z0.h, #0
// CHECK-ENCODING: [0x00,0xac,0x10,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4510ac00 <unknown>

ushllt     z31.s, z31.h, #15
// CHECK-INST: ushllt	z31.s, z31.h, #15
// CHECK-ENCODING: [0xff,0xaf,0x1f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 451fafff <unknown>

ushllt     z0.d, z0.s, #0
// CHECK-INST: ushllt	z0.d, z0.s, #0
// CHECK-ENCODING: [0x00,0xac,0x40,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 4540ac00 <unknown>

ushllt     z31.d, z31.s, #31
// CHECK-INST: ushllt	z31.d, z31.s, #31
// CHECK-ENCODING: [0xff,0xaf,0x5f,0x45]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 455fafff <unknown>
