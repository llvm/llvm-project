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

sqincp  x0, p0.b
// CHECK-INST: sqincp  x0, p0.b
// CHECK-ENCODING: [0x00,0x8c,0x28,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25288c00 <unknown>

sqincp  x0, p0.h
// CHECK-INST: sqincp  x0, p0.h
// CHECK-ENCODING: [0x00,0x8c,0x68,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25688c00 <unknown>

sqincp  x0, p0.s
// CHECK-INST: sqincp  x0, p0.s
// CHECK-ENCODING: [0x00,0x8c,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a88c00 <unknown>

sqincp  x0, p0.d
// CHECK-INST: sqincp  x0, p0.d
// CHECK-ENCODING: [0x00,0x8c,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e88c00 <unknown>

sqincp  xzr, p15.b, wzr
// CHECK-INST: sqincp  xzr, p15.b, wzr
// CHECK-ENCODING: [0xff,0x89,0x28,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 252889ff <unknown>

sqincp  xzr, p15.h, wzr
// CHECK-INST: sqincp  xzr, p15.h, wzr
// CHECK-ENCODING: [0xff,0x89,0x68,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256889ff <unknown>

sqincp  xzr, p15.s, wzr
// CHECK-INST: sqincp  xzr, p15.s, wzr
// CHECK-ENCODING: [0xff,0x89,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a889ff <unknown>

sqincp  xzr, p15.d, wzr
// CHECK-INST: sqincp  xzr, p15.d, wzr
// CHECK-ENCODING: [0xff,0x89,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e889ff <unknown>

sqincp  z0.h, p0
// CHECK-INST: sqincp  z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x68,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25688000 <unknown>

sqincp  z0.h, p0.h
// CHECK-INST: sqincp  z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x68,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25688000 <unknown>

sqincp  z0.s, p0
// CHECK-INST: sqincp  z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a88000 <unknown>

sqincp  z0.s, p0.s
// CHECK-INST: sqincp  z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a88000 <unknown>

sqincp  z0.d, p0
// CHECK-INST: sqincp  z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e88000 <unknown>

sqincp  z0.d, p0.d
// CHECK-INST: sqincp  z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e88000 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqincp  z0.d, p0.d
// CHECK-INST: sqincp	z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e88000 <unknown>
