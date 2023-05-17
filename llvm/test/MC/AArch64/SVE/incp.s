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

incp    x0, p0.b
// CHECK-INST: incp    x0, p0.b
// CHECK-ENCODING: [0x00,0x88,0x2c,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 252c8800 <unknown>

incp    x0, p0.h
// CHECK-INST: incp    x0, p0.h
// CHECK-ENCODING: [0x00,0x88,0x6c,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256c8800 <unknown>

incp    x0, p0.s
// CHECK-INST: incp    x0, p0.s
// CHECK-ENCODING: [0x00,0x88,0xac,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ac8800 <unknown>

incp    x0, p0.d
// CHECK-INST: incp    x0, p0.d
// CHECK-ENCODING: [0x00,0x88,0xec,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ec8800 <unknown>

incp    xzr, p15.b
// CHECK-INST: incp    xzr, p15.b
// CHECK-ENCODING: [0xff,0x89,0x2c,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 252c89ff <unknown>

incp    xzr, p15.h
// CHECK-INST: incp    xzr, p15.h
// CHECK-ENCODING: [0xff,0x89,0x6c,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256c89ff <unknown>

incp    xzr, p15.s
// CHECK-INST: incp    xzr, p15.s
// CHECK-ENCODING: [0xff,0x89,0xac,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ac89ff <unknown>

incp    xzr, p15.d
// CHECK-INST: incp    xzr, p15.d
// CHECK-ENCODING: [0xff,0x89,0xec,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ec89ff <unknown>

incp    z31.h, p15
// CHECK-INST: incp    z31.h, p15.h
// CHECK-ENCODING: [0xff,0x81,0x6c,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256c81ff <unknown>

incp    z31.h, p15.h
// CHECK-INST: incp    z31.h, p15.h
// CHECK-ENCODING: [0xff,0x81,0x6c,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256c81ff <unknown>

incp    z31.s, p15
// CHECK-INST: incp    z31.s, p15.s
// CHECK-ENCODING: [0xff,0x81,0xac,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ac81ff <unknown>

incp    z31.s, p15.s
// CHECK-INST: incp    z31.s, p15.s
// CHECK-ENCODING: [0xff,0x81,0xac,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ac81ff <unknown>

incp    z31.d, p15
// CHECK-INST: incp    z31.d, p15.d
// CHECK-ENCODING: [0xff,0x81,0xec,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ec81ff <unknown>

incp    z31.d, p15.d
// CHECK-INST: incp    z31.d, p15.d
// CHECK-ENCODING: [0xff,0x81,0xec,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ec81ff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

incp    z31.d, p15.d
// CHECK-INST: incp	z31.d, p15.d
// CHECK-ENCODING: [0xff,0x81,0xec,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ec81ff <unknown>
