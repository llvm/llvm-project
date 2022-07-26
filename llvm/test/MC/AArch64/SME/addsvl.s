// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:   | llvm-objdump -d --mattr=-sme - | FileCheck %s --check-prefix=CHECK-UNKNOWN

addsvl   x21, x21, #0
// CHECK-INST: addsvl   x21, x21, #0
// CHECK-ENCODING: [0x15,0x58,0x35,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 04355815 <unknown>

addsvl   x23, x8, #-1
// CHECK-INST: addsvl   x23, x8, #-1
// CHECK-ENCODING: [0xf7,0x5f,0x28,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 04285ff7 <unknown>

addsvl   sp, sp, #31
// CHECK-INST: addsvl   sp, sp, #31
// CHECK-ENCODING: [0xff,0x5b,0x3f,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 043f5bff <unknown>

addsvl   x0, x0, #-32
// CHECK-INST: addsvl   x0, x0, #-32
// CHECK-ENCODING: [0x00,0x5c,0x20,0x04]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 04205c00 <unknown>
