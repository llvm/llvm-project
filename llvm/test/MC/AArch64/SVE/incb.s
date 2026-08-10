// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

incb    x0
// CHECK-INST: incb    x0
// CHECK-ENCODING: [0xe0,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e3e0 <unknown>

incb    x0, all
// CHECK-INST: incb    x0
// CHECK-ENCODING: [0xe0,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e3e0 <unknown>

incb    x0, all, mul #1
// CHECK-INST: incb    x0
// CHECK-ENCODING: [0xe0,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e3e0 <unknown>

incb    x0, all, mul #16
// CHECK-INST: incb    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe3,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043fe3e0 <unknown>

incb    x0, pow2
// CHECK-INST: incb    x0, pow2
// CHECK-ENCODING: [0x00,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e000 <unknown>

incb    x0, vl1
// CHECK-INST: incb    x0, vl1
// CHECK-ENCODING: [0x20,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e020 <unknown>

incb    x0, vl2
// CHECK-INST: incb    x0, vl2
// CHECK-ENCODING: [0x40,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e040 <unknown>

incb    x0, vl3
// CHECK-INST: incb    x0, vl3
// CHECK-ENCODING: [0x60,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e060 <unknown>

incb    x0, vl4
// CHECK-INST: incb    x0, vl4
// CHECK-ENCODING: [0x80,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e080 <unknown>

incb    x0, vl5
// CHECK-INST: incb    x0, vl5
// CHECK-ENCODING: [0xa0,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e0a0 <unknown>

incb    x0, vl6
// CHECK-INST: incb    x0, vl6
// CHECK-ENCODING: [0xc0,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e0c0 <unknown>

incb    x0, vl7
// CHECK-INST: incb    x0, vl7
// CHECK-ENCODING: [0xe0,0xe0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e0e0 <unknown>

incb    x0, vl8
// CHECK-INST: incb    x0, vl8
// CHECK-ENCODING: [0x00,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e100 <unknown>

incb    x0, vl16
// CHECK-INST: incb    x0, vl16
// CHECK-ENCODING: [0x20,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e120 <unknown>

incb    x0, vl32
// CHECK-INST: incb    x0, vl32
// CHECK-ENCODING: [0x40,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e140 <unknown>

incb    x0, vl64
// CHECK-INST: incb    x0, vl64
// CHECK-ENCODING: [0x60,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e160 <unknown>

incb    x0, vl128
// CHECK-INST: incb    x0, vl128
// CHECK-ENCODING: [0x80,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e180 <unknown>

incb    x0, vl256
// CHECK-INST: incb    x0, vl256
// CHECK-ENCODING: [0xa0,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e1a0 <unknown>

incb    x0, #14
// CHECK-INST: incb    x0, #14
// CHECK-ENCODING: [0xc0,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e1c0 <unknown>

incb    x0, #15
// CHECK-INST: incb    x0, #15
// CHECK-ENCODING: [0xe0,0xe1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e1e0 <unknown>

incb    x0, #16
// CHECK-INST: incb    x0, #16
// CHECK-ENCODING: [0x00,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e200 <unknown>

incb    x0, #17
// CHECK-INST: incb    x0, #17
// CHECK-ENCODING: [0x20,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e220 <unknown>

incb    x0, #18
// CHECK-INST: incb    x0, #18
// CHECK-ENCODING: [0x40,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e240 <unknown>

incb    x0, #19
// CHECK-INST: incb    x0, #19
// CHECK-ENCODING: [0x60,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e260 <unknown>

incb    x0, #20
// CHECK-INST: incb    x0, #20
// CHECK-ENCODING: [0x80,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e280 <unknown>

incb    x0, #21
// CHECK-INST: incb    x0, #21
// CHECK-ENCODING: [0xa0,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e2a0 <unknown>

incb    x0, #22
// CHECK-INST: incb    x0, #22
// CHECK-ENCODING: [0xc0,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e2c0 <unknown>

incb    x0, #23
// CHECK-INST: incb    x0, #23
// CHECK-ENCODING: [0xe0,0xe2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e2e0 <unknown>

incb    x0, #24
// CHECK-INST: incb    x0, #24
// CHECK-ENCODING: [0x00,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e300 <unknown>

incb    x0, #25
// CHECK-INST: incb    x0, #25
// CHECK-ENCODING: [0x20,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e320 <unknown>

incb    x0, #26
// CHECK-INST: incb    x0, #26
// CHECK-ENCODING: [0x40,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e340 <unknown>

incb    x0, #27
// CHECK-INST: incb    x0, #27
// CHECK-ENCODING: [0x60,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e360 <unknown>

incb    x0, #28
// CHECK-INST: incb    x0, #28
// CHECK-ENCODING: [0x80,0xe3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430e380 <unknown>
