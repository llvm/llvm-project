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

dech    x0
// CHECK-INST: dech    x0
// CHECK-ENCODING: [0xe0,0xe7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e7e0 <unknown>

dech    x0, all
// CHECK-INST: dech    x0
// CHECK-ENCODING: [0xe0,0xe7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e7e0 <unknown>

dech    x0, all, mul #1
// CHECK-INST: dech    x0
// CHECK-ENCODING: [0xe0,0xe7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e7e0 <unknown>

dech    x0, all, mul #16
// CHECK-INST: dech    x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xe7,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 047fe7e0 <unknown>

dech    x0, pow2
// CHECK-INST: dech    x0, pow2
// CHECK-ENCODING: [0x00,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e400 <unknown>

dech    x0, vl1
// CHECK-INST: dech    x0, vl1
// CHECK-ENCODING: [0x20,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e420 <unknown>

dech    x0, vl2
// CHECK-INST: dech    x0, vl2
// CHECK-ENCODING: [0x40,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e440 <unknown>

dech    x0, vl3
// CHECK-INST: dech    x0, vl3
// CHECK-ENCODING: [0x60,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e460 <unknown>

dech    x0, vl4
// CHECK-INST: dech    x0, vl4
// CHECK-ENCODING: [0x80,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e480 <unknown>

dech    x0, vl5
// CHECK-INST: dech    x0, vl5
// CHECK-ENCODING: [0xa0,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e4a0 <unknown>

dech    x0, vl6
// CHECK-INST: dech    x0, vl6
// CHECK-ENCODING: [0xc0,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e4c0 <unknown>

dech    x0, vl7
// CHECK-INST: dech    x0, vl7
// CHECK-ENCODING: [0xe0,0xe4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e4e0 <unknown>

dech    x0, vl8
// CHECK-INST: dech    x0, vl8
// CHECK-ENCODING: [0x00,0xe5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e500 <unknown>

dech    x0, vl16
// CHECK-INST: dech    x0, vl16
// CHECK-ENCODING: [0x20,0xe5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e520 <unknown>

dech    x0, vl32
// CHECK-INST: dech    x0, vl32
// CHECK-ENCODING: [0x40,0xe5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e540 <unknown>

dech    x0, vl64
// CHECK-INST: dech    x0, vl64
// CHECK-ENCODING: [0x60,0xe5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e560 <unknown>

dech    x0, vl128
// CHECK-INST: dech    x0, vl128
// CHECK-ENCODING: [0x80,0xe5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e580 <unknown>

dech    x0, vl256
// CHECK-INST: dech    x0, vl256
// CHECK-ENCODING: [0xa0,0xe5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e5a0 <unknown>

dech    x0, #14
// CHECK-INST: dech    x0, #14
// CHECK-ENCODING: [0xc0,0xe5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e5c0 <unknown>

dech    x0, #28
// CHECK-INST: dech    x0, #28
// CHECK-ENCODING: [0x80,0xe7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470e780 <unknown>
