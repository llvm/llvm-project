// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

not     z31.b, p7/m, z31.b
// CHECK-INST: not	z31.b, p7/m, z31.b
// CHECK-ENCODING: [0xff,0xbf,0x1e,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 041ebfff <unknown>

not     z31.h, p7/m, z31.h
// CHECK-INST: not	z31.h, p7/m, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x5e,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 045ebfff <unknown>

not     z31.s, p7/m, z31.s
// CHECK-INST: not	z31.s, p7/m, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x9e,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 049ebfff <unknown>

not     z31.d, p7/m, z31.d
// CHECK-INST: not	z31.d, p7/m, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xde,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04debfff <unknown>

not     p0.b, p0/z, p0.b
// CHECK-INST: not     p0.b, p0/z, p0.b
// CHECK-ENCODING: [0x00,0x42,0x00,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25004200 <unknown>

not     p15.b, p15/z, p15.b
// CHECK-INST: not     p15.b, p15/z, p15.b
// CHECK-ENCODING: [0xef,0x7f,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 250f7fef <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cc4 <unknown>

not     z4.d, p7/m, z31.d
// CHECK-INST: not	z4.d, p7/m, z31.d
// CHECK-ENCODING: [0xe4,0xbf,0xde,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04debfe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

not     z4.d, p7/m, z31.d
// CHECK-INST: not	z4.d, p7/m, z31.d
// CHECK-ENCODING: [0xe4,0xbf,0xde,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04debfe4 <unknown>
