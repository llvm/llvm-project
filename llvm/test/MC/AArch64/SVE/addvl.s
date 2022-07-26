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

addvl   x21, x21, #0
// CHECK-INST: addvl   x21, x21, #0
// CHECK-ENCODING: [0x15,0x50,0x35,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04355015 <unknown>

addvl   x23, x8, #-1
// CHECK-INST: addvl   x23, x8, #-1
// CHECK-ENCODING: [0xf7,0x57,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042857f7 <unknown>

addvl   sp, sp, #31
// CHECK-INST: addvl   sp, sp, #31
// CHECK-ENCODING: [0xff,0x53,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043f53ff <unknown>

addvl   x0, x0, #-32
// CHECK-INST: addvl   x0, x0, #-32
// CHECK-ENCODING: [0x00,0x54,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04205400 <unknown>
