// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3 < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2p3 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p3 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// ---------------------------------------------------------------
// Lookup table read with 6-bit indices (8-bit)

luti6 z0.b, { z0.b, z1.b }, z0
// CHECK-INST: luti6 z0.b, { z0.b, z1.b }, z0
// CHECK-ENCODING: encoding: [0x00,0xac,0x20,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 4520ac00 <unknown>

luti6 z10.b, { z0.b, z1.b }, z0
// CHECK-INST: luti6 z10.b, { z0.b, z1.b }, z0
// CHECK-ENCODING: encoding: [0x0a,0xac,0x20,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 4520ac0a <unknown>

luti6 z21.b, { z0.b, z1.b }, z0
// CHECK-INST: luti6 z21.b, { z0.b, z1.b }, z0
// CHECK-ENCODING: encoding: [0x15,0xac,0x20,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 4520ac15 <unknown>

luti6 z31.b, { z0.b, z1.b }, z0
// CHECK-INST: luti6 z31.b, { z0.b, z1.b }, z0
// CHECK-ENCODING: encoding: [0x1f,0xac,0x20,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 4520ac1f <unknown>

luti6 z0.b, { z31.b, z0.b }, z31
// CHECK-INST: luti6 z0.b, { z31.b, z0.b }, z31
// CHECK-ENCODING: encoding: [0xe0,0xaf,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 453fafe0 <unknown>

luti6 z10.b, { z31.b, z0.b }, z31
// CHECK-INST: luti6 z10.b, { z31.b, z0.b }, z31
// CHECK-ENCODING: encoding: [0xea,0xaf,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 453fafea <unknown>

luti6 z21.b, { z31.b, z0.b }, z31
// CHECK-INST: luti6 z21.b, { z31.b, z0.b }, z31
// CHECK-ENCODING: encoding: [0xf5,0xaf,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 453faff5 <unknown>

luti6 z31.b, { z31.b, z0.b }, z31
// CHECK-INST: luti6 z31.b, { z31.b, z0.b }, z31
// CHECK-ENCODING: encoding: [0xff,0xaf,0x3f,0x45]
// CHECK-ERROR: instruction requires: sve2p3
// CHECK-UNKNOWN: 453fafff <unknown>

// ---------------------------------------------------------------
// Lookup table read with 6-bit indices (16-bit)

luti6 z0.h, { z0.h, z1.h }, z0[0]
// CHECK-INST: luti6 z0.h, { z0.h, z1.h }, z0[0]
// CHECK-ENCODING: encoding: [0x00,0xac,0x60,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4560ac00 <unknown>

luti6 z10.h, { z0.h, z1.h }, z0[0]
// CHECK-INST: luti6 z10.h, { z0.h, z1.h }, z0[0]
// CHECK-ENCODING: encoding: [0x0a,0xac,0x60,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4560ac0a <unknown>

luti6 z21.h, { z0.h, z1.h }, z0[0]
// CHECK-INST: luti6 z21.h, { z0.h, z1.h }, z0[0]
// CHECK-ENCODING: encoding: [0x15,0xac,0x60,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4560ac15 <unknown>

luti6 z31.h, { z0.h, z1.h }, z0[0]
// CHECK-INST: luti6 z31.h, { z0.h, z1.h }, z0[0]
// CHECK-ENCODING: encoding: [0x1f,0xac,0x60,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 4560ac1f <unknown>

luti6 z0.h, { z31.h, z0.h }, z31[1]
// CHECK-INST: luti6 z0.h, { z31.h, z0.h }, z31[1]
// CHECK-ENCODING: encoding: [0xe0,0xaf,0xff,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45ffafe0 <unknown>

luti6 z10.h, { z31.h, z0.h }, z31[1]
// CHECK-INST: luti6 z10.h, { z31.h, z0.h }, z31[1]
// CHECK-ENCODING: encoding: [0xea,0xaf,0xff,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45ffafea <unknown>

luti6 z21.h, { z31.h, z0.h }, z31[1]
// CHECK-INST: luti6 z21.h, { z31.h, z0.h }, z31[1]
// CHECK-ENCODING: encoding: [0xf5,0xaf,0xff,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45ffaff5 <unknown>

luti6 z31.h, { z31.h, z0.h }, z31[1]
// CHECK-INST: luti6 z31.h, { z31.h, z0.h }, z31[1]
// CHECK-ENCODING: encoding: [0xff,0xaf,0xff,0x45]
// CHECK-ERROR: instruction requires: sme2p3 or sve2p3
// CHECK-UNKNOWN: 45ffafff <unknown>
