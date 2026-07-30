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

decp    x0, p0.b
// CHECK-INST: decp    x0, p0.b
// CHECK-ENCODING: [0x00,0x88,0x2d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 252d8800 <unknown>

decp    x0, p0.h
// CHECK-INST: decp    x0, p0.h
// CHECK-ENCODING: [0x00,0x88,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256d8800 <unknown>

decp    x0, p0.s
// CHECK-INST: decp    x0, p0.s
// CHECK-ENCODING: [0x00,0x88,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ad8800 <unknown>

decp    x0, p0.d
// CHECK-INST: decp    x0, p0.d
// CHECK-ENCODING: [0x00,0x88,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ed8800 <unknown>

decp    xzr, p15.b
// CHECK-INST: decp    xzr, p15.b
// CHECK-ENCODING: [0xff,0x89,0x2d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 252d89ff <unknown>

decp    xzr, p15.h
// CHECK-INST: decp    xzr, p15.h
// CHECK-ENCODING: [0xff,0x89,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256d89ff <unknown>

decp    xzr, p15.s
// CHECK-INST: decp    xzr, p15.s
// CHECK-ENCODING: [0xff,0x89,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ad89ff <unknown>

decp    xzr, p15.d
// CHECK-INST: decp    xzr, p15.d
// CHECK-ENCODING: [0xff,0x89,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ed89ff <unknown>

decp    z31.h, p15
// CHECK-INST: decp    z31.h, p15.h
// CHECK-ENCODING: [0xff,0x81,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256d81ff <unknown>

decp    z31.h, p15.h
// CHECK-INST: decp    z31.h, p15.h
// CHECK-ENCODING: [0xff,0x81,0x6d,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 256d81ff <unknown>

decp    z31.s, p15
// CHECK-INST: decp    z31.s, p15.s
// CHECK-ENCODING: [0xff,0x81,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ad81ff <unknown>

decp    z31.s, p15.s
// CHECK-INST: decp    z31.s, p15.s
// CHECK-ENCODING: [0xff,0x81,0xad,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ad81ff <unknown>

decp    z31.d, p15
// CHECK-INST: decp    z31.d, p15.d
// CHECK-ENCODING: [0xff,0x81,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ed81ff <unknown>

decp    z31.d, p15.d
// CHECK-INST: decp    z31.d, p15.d
// CHECK-ENCODING: [0xff,0x81,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ed81ff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

decp    z31.d, p15.d
// CHECK-INST: decp	z31.d, p15
// CHECK-ENCODING: [0xff,0x81,0xed,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25ed81ff <unknown>
