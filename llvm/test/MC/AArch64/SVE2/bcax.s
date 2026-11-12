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

bcax z29.d, z29.d, z30.d, z31.d
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 047e3bfd <unknown>


// --------------------------------------------------------------------------//
// Test aliases.

bcax z29.b, z29.b, z30.b, z31.b
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 047e3bfd <unknown>

bcax z29.h, z29.h, z30.h, z31.h
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 047e3bfd <unknown>

bcax z29.s, z29.s, z30.s, z31.s
// CHECK-INST: bcax z29.d, z29.d, z30.d, z31.d
// CHECK-ENCODING: [0xfd,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 047e3bfd <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z7
// CHECK-INST: movprfx z31, z7
// CHECK-ENCODING: [0xff,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcff <unknown>

bcax z31.d, z31.d, z30.d, z29.d
// CHECK-INST: bcax z31.d, z31.d, z30.d, z29.d
// CHECK-ENCODING: [0xbf,0x3b,0x7e,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 047e3bbf <unknown>
