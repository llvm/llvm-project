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

and     z5.b, z5.b, #0xf9
// CHECK-INST: and     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x80,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05802ea5 <unknown>

and     z23.h, z23.h, #0xfff9
// CHECK-INST: and     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x80,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05806db7 <unknown>

and     z0.s, z0.s, #0xfffffff9
// CHECK-INST: and     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x80,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0580eba0 <unknown>

and     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-INST: and     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x83,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0583efa0 <unknown>

and     z5.b, z5.b, #0x6
// CHECK-INST: and     z5.b, z5.b, #0x6
// CHECK-ENCODING: [0x25,0x3e,0x80,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05803e25 <unknown>

and     z23.h, z23.h, #0x6
// CHECK-INST: and     z23.h, z23.h, #0x6
// CHECK-ENCODING: [0x37,0x7c,0x80,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05807c37 <unknown>

and     z0.s, z0.s, #0x6
// CHECK-INST: and     z0.s, z0.s, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x80,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0580f820 <unknown>

and     z0.d, z0.d, #0x6
// CHECK-INST: and     z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x83,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0583f820 <unknown>

and     z0.d, z0.d, z0.d
// CHECK-INST: and     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04203000 <unknown>

and     z23.d, z13.d, z8.d
// CHECK-INST: and     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x31,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042831b7 <unknown>

and     z31.b, p7/m, z31.b, z31.b
// CHECK-INST: and     z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x1a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 041a1fff <unknown>

and     z31.h, p7/m, z31.h, z31.h
// CHECK-INST: and     z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x5a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 045a1fff <unknown>

and     z31.s, p7/m, z31.s, z31.s
// CHECK-INST: and     z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x9a,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 049a1fff <unknown>

and     z31.d, p7/m, z31.d, z31.d
// CHECK-INST: and     z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xda,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04da1fff <unknown>

and     p0.b, p0/z, p0.b, p1.b
// CHECK-INST: and     p0.b, p0/z, p0.b, p1.b
// CHECK-ENCODING: [0x00,0x40,0x01,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25014000 <unknown>

and     p0.b, p0/z, p0.b, p0.b
// CHECK-INST: mov     p0.b, p0/z, p0.b
// CHECK-ENCODING: [0x00,0x40,0x00,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25004000 <unknown>

and     p15.b, p15/z, p15.b, p15.b
// CHECK-INST: mov     p15.b, p15/z, p15.b
// CHECK-ENCODING: [0xef,0x7d,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 250f7def <unknown>


// --------------------------------------------------------------------------//
// Test aliases.

and     z0.s, z0.s, z0.s
// CHECK-INST: and     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04203000 <unknown>

and     z0.h, z0.h, z0.h
// CHECK-INST: and     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04203000 <unknown>

and     z0.b, z0.b, z0.b
// CHECK-INST: and     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04203000 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cc4 <unknown>

and     z4.d, p7/m, z4.d, z31.d
// CHECK-INST: and	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xda,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04da1fe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

and     z4.d, p7/m, z4.d, z31.d
// CHECK-INST: and	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xda,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04da1fe4 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

and     z0.d, z0.d, #0x6
// CHECK-INST: and	z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x83,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0583f820 <unknown>
