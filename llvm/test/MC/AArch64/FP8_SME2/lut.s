// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme2p1,+sme-lutv2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme2p1,+sme-lutv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme2p1,+sme-lutv2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme2p1,+sme-lutv2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-lutv2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme2p1,+sme-lutv2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme2p1,+sme-lutv2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

luti4   {z0.b-z3.b}, zt0, {z0-z1}  // 11000000-10001011-00000000-00000000
// CHECK-INST: luti4   { z0.b - z3.b }, zt0, { z0, z1 }
// CHECK-ENCODING: [0x00,0x00,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2 sme-lutv2
// CHECK-UNKNOWN: c08b0000 <unknown>

luti4   {z28.b-z31.b}, zt0, {z30-z31}  // 11000000-10001011-00000011-11011100
// CHECK-INST: luti4   { z28.b - z31.b }, zt0, { z30, z31 }
// CHECK-ENCODING: [0xdc,0x03,0x8b,0xc0]
// CHECK-ERROR: instruction requires: sme2 sme-lutv2
// CHECK-UNKNOWN: c08b03dc <unknown>

// Strided
luti4   {z0.b, z4.b, z8.b, z12.b}, zt0, {z0-z1}  // 11000000-10011011-00000000-00000000
// CHECK-INST: luti4   { z0.b, z4.b, z8.b, z12.b }, zt0, { z0, z1 }
// CHECK-ENCODING: [0x00,0x00,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1 sme-lutv2
// CHECK-UNKNOWN: c09b0000 <unknown>

luti4   {z19.b, z23.b, z27.b, z31.b}, zt0, {z30-z31}  // 11000000-10011011-00000011-11010011
// CHECK-INST: luti4   { z19.b, z23.b, z27.b, z31.b }, zt0, { z30, z31 }
// CHECK-ENCODING: [0xd3,0x03,0x9b,0xc0]
// CHECK-ERROR: instruction requires: sme2p1 sme-lutv2
// CHECK-UNKNOWN: c09b03d3 <unknown>
