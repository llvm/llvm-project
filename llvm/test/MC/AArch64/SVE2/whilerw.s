// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

whilerw  p15.b, x30, x30
// CHECK-INST: whilerw  p15.b, x30, x30
// CHECK-ENCODING: [0xdf,0x33,0x3e,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 33 3e 25 <unknown>

whilerw  p15.h, x30, x30
// CHECK-INST: whilerw  p15.h, x30, x30
// CHECK-ENCODING: [0xdf,0x33,0x7e,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 33 7e 25 <unknown>

whilerw  p15.s, x30, x30
// CHECK-INST: whilerw  p15.s, x30, x30
// CHECK-ENCODING: [0xdf,0x33,0xbe,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 33 be 25 <unknown>

whilerw  p15.d, x30, x30
// CHECK-INST: whilerw  p15.d, x30, x30
// CHECK-ENCODING: [0xdf,0x33,0xfe,0x25]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: df 33 fe 25 <unknown>
