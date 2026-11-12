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

flogb    z31.h, p7/m, z31.h
// CHECK-INST: flogb	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x1a,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 651abfff <unknown>

flogb    z31.s, p7/m, z31.s
// CHECK-INST: flogb	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x1c,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 651cbfff <unknown>

flogb    z31.d, p7/m, z31.d
// CHECK-INST: flogb	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0x1e,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 651ebfff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cc4 <unknown>

flogb    z4.d, p7/m, z31.d
// CHECK-INST: flogb	z4.d, p7/m, z31.d
// CHECK-ENCODING: [0xe4,0xbf,0x1e,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 651ebfe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

flogb    z4.d, p7/m, z31.d
// CHECK-INST: flogb	z4.d, p7/m, z31.d
// CHECK-ENCODING: [0xe4,0xbf,0x1e,0x65]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 651ebfe4 <unknown>
