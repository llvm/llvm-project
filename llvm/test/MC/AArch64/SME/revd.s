// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

revd    z0.q, p0/m, z0.q
// CHECK-INST: revd    z0.q, p0/m, z0.q
// CHECK-ENCODING: [0x00,0x80,0x2e,0x05]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 052e8000 <unknown>

revd    z21.q, p5/m, z10.q
// CHECK-INST: revd    z21.q, p5/m, z10.q
// CHECK-ENCODING: [0x55,0x95,0x2e,0x05]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 052e9555 <unknown>

revd    z23.q, p3/m, z13.q
// CHECK-INST: revd    z23.q, p3/m, z13.q
// CHECK-ENCODING: [0xb7,0x8d,0x2e,0x05]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 052e8db7 <unknown>

revd    z31.q, p7/m, z31.q
// CHECK-INST: revd    z31.q, p7/m, z31.q
// CHECK-ENCODING: [0xff,0x9f,0x2e,0x05]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 052e9fff <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z21, z25
// CHECK-INST: movprfx  z21, z25
// CHECK-ENCODING: [0x35,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf35 <unknown>

revd    z21.q, p5/m, z10.q
// CHECK-INST: revd    z21.q, p5/m, z10.q
// CHECK-ENCODING: [0x55,0x95,0x2e,0x05]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 052e9555 <unknown>
