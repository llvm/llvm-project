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


umlslb z0.h, z1.b, z31.b
// CHECK-INST: umlslb	z0.h, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x58,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 445f5820 <unknown>

umlslb z0.s, z1.h, z31.h
// CHECK-INST: umlslb	z0.s, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x58,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 449f5820 <unknown>

umlslb z0.d, z1.s, z31.s
// CHECK-INST: umlslb	z0.d, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x58,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df5820 <unknown>

umlslb z0.s, z1.h, z7.h[7]
// CHECK-INST: umlslb	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xb8,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44bfb820 <unknown>

umlslb z0.d, z1.s, z15.s[1]
// CHECK-INST: umlslb	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0xb8,0xef,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44efb820 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

umlslb z21.d, z1.s, z31.s
// CHECK-INST: umlslb	z21.d, z1.s, z31.s
// CHECK-ENCODING: [0x35,0x58,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df5835 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

umlslb   z21.d, z10.s, z5.s[1]
// CHECK-INST: umlslb   z21.d, z10.s, z5.s[1]
// CHECK-ENCODING: [0x55,0xb9,0xe5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44e5b955 <unknown>
