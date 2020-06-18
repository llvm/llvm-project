// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

bcax z29.d, z29.d, z30.d, z31.d
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: fd 3b 7e 04 <unknown>


// --------------------------------------------------------------------------//
// Test aliases.

bcax z29.b, z29.b, z30.b, z31.b
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: fd 3b 7e 04 <unknown>

bcax z29.h, z29.h, z30.h, z31.h
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: fd 3b 7e 04 <unknown>

bcax z29.s, z29.s, z30.s, z31.s
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: fd 3b 7e 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z7
// CHECK-INST: movprfx z31, z7
// CHECK-ENCODING: [0xff,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff bc 20 04 <unknown>

bcax z31.d, z31.d, z30.d, z29.d
// CHECK-INST: bcax z31.d, z31.d, z30.d, z29.d
// CHECK-ENCODING: [0xbf,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: bf 3b 7e 04 <unknown>
