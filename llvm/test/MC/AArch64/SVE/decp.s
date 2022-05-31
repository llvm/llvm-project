// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

decp    x0, p0.b
// CHECK-INST: decp    x0, p0.b
// CHECK-ENCODING: [0x00,0x88,0x2d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 88 2d 25 <unknown>

decp    x0, p0.h
// CHECK-INST: decp    x0, p0.h
// CHECK-ENCODING: [0x00,0x88,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 88 6d 25 <unknown>

decp    x0, p0.s
// CHECK-INST: decp    x0, p0.s
// CHECK-ENCODING: [0x00,0x88,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 88 ad 25 <unknown>

decp    x0, p0.d
// CHECK-INST: decp    x0, p0.d
// CHECK-ENCODING: [0x00,0x88,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 88 ed 25 <unknown>

decp    xzr, p15.b
// CHECK-INST: decp    xzr, p15.b
// CHECK-ENCODING: [0xff,0x89,0x2d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 2d 25 <unknown>

decp    xzr, p15.h
// CHECK-INST: decp    xzr, p15.h
// CHECK-ENCODING: [0xff,0x89,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 6d 25 <unknown>

decp    xzr, p15.s
// CHECK-INST: decp    xzr, p15.s
// CHECK-ENCODING: [0xff,0x89,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 ad 25 <unknown>

decp    xzr, p15.d
// CHECK-INST: decp    xzr, p15.d
// CHECK-ENCODING: [0xff,0x89,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 ed 25 <unknown>

decp    z31.h, p15
// CHECK-INST: decp    z31.h, p15.h
// CHECK-ENCODING: [0xff,0x81,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 6d 25 <unknown>

decp    z31.h, p15.h
// CHECK-INST: decp    z31.h, p15.h
// CHECK-ENCODING: [0xff,0x81,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 6d 25 <unknown>

decp    z31.s, p15
// CHECK-INST: decp    z31.s, p15.s
// CHECK-ENCODING: [0xff,0x81,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 ad 25 <unknown>

decp    z31.s, p15.s
// CHECK-INST: decp    z31.s, p15.s
// CHECK-ENCODING: [0xff,0x81,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 ad 25 <unknown>

decp    z31.d, p15
// CHECK-INST: decp    z31.d, p15.d
// CHECK-ENCODING: [0xff,0x81,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 ed 25 <unknown>

decp    z31.d, p15.d
// CHECK-INST: decp    z31.d, p15.d
// CHECK-ENCODING: [0xff,0x81,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 ed 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

decp    z31.d, p15.d
// CHECK-INST: decp	z31.d, p15
// CHECK-ENCODING: [0xff,0x81,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 81 ed 25 <unknown>
