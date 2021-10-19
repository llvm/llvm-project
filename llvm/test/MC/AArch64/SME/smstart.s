// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d --mattr=+sme - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

smstart
// CHECK-INST: smstart
// CHECK-ENCODING: [0x7f,0x47,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 47 03 d5   msr   S0_3_C4_C7_3, xzr

smstart sm
// CHECK-INST: smstart sm
// CHECK-ENCODING: [0x7f,0x43,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 43 03 d5   msr   S0_3_C4_C3_3, xzr

smstart za
// CHECK-INST: smstart za
// CHECK-ENCODING: [0x7f,0x45,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 45 03 d5   msr   S0_3_C4_C5_3, xzr

smstart SM
// CHECK-INST: smstart sm
// CHECK-ENCODING: [0x7f,0x43,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 43 03 d5   msr   S0_3_C4_C3_3, xzr

smstart ZA
// CHECK-INST: smstart za
// CHECK-ENCODING: [0x7f,0x45,0x03,0xd5]
// CHECK-ERROR: instruction requires: sme
// CHECK-UNKNOWN: 7f 45 03 d5   msr   S0_3_C4_C5_3, xzr
