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

eon     z5.b, z5.b, #0xf9
// CHECK-INST: eor     z5.b, z5.b, #0x6
// CHECK-ENCODING: [0x25,0x3e,0x40,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05403e25 <unknown>

eon     z23.h, z23.h, #0xfff9
// CHECK-INST: eor     z23.h, z23.h, #0x6
// CHECK-ENCODING: [0x37,0x7c,0x40,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05407c37 <unknown>

eon     z0.s, z0.s, #0xfffffff9
// CHECK-INST: eor     z0.s, z0.s, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x40,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0540f820 <unknown>

eon     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-INST: eor     z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x43,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0543f820 <unknown>

eon     z5.b, z5.b, #0x6
// CHECK-INST: eor     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x40,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05402ea5 <unknown>

eon     z23.h, z23.h, #0x6
// CHECK-INST: eor     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x40,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05406db7 <unknown>

eon     z0.s, z0.s, #0x6
// CHECK-INST: eor     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x40,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0540eba0 <unknown>

eon     z0.d, z0.d, #0x6
// CHECK-INST: eor     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x43,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0543efa0 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

eon     z0.d, z0.d, #0x6
// CHECK-INST: eor	z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x43,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0543efa0 <unknown>
