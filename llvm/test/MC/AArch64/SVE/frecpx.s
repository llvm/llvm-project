// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

frecpx   z31.h, p7/m, z31.h
// CHECK-INST: frecpx	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x4c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf 4c 65 <unknown>

frecpx   z31.s, p7/m, z31.s
// CHECK-INST: frecpx	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x8c,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf 8c 65 <unknown>

frecpx   z31.d, p7/m, z31.d
// CHECK-INST: frecpx	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xcc,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff bf cc 65 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 3c d0 04 <unknown>

frecpx   z4.d, p7/m, z31.d
// CHECK-INST: frecpx	z4.d, p7/m, z31.d
// CHECK-ENCODING: [0xe4,0xbf,0xcc,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 bf cc 65 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

frecpx   z4.d, p7/m, z31.d
// CHECK-INST: frecpx	z4.d, p7/m, z31.d
// CHECK-ENCODING: [0xe4,0xbf,0xcc,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 bf cc 65 <unknown>
