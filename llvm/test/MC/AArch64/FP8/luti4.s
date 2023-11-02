// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lut < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lut < %s \
// RUN:        | llvm-objdump -d --mattr=+lut - | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lut < %s \
// RUN:        | llvm-objdump -d --mattr=-lut - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lut < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+lut -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

luti4   v1.16b, {v2.16b}, v0[0]  // 01001110-01000000-00100000-01000001
// CHECK-INST: luti4   v1.16b, { v2.16b }, v0[0]
// CHECK-ENCODING: [0x41,0x20,0x40,0x4e]
// CHECK-ERROR: instruction requires: lut
// CHECK-UNKNOWN: 4e402041 <unknown>

luti4   v30.16b, {v20.16b}, v31[1]  // 01001110-01011111-01100010-10011110
// CHECK-INST: luti4   v30.16b, { v20.16b }, v31[1]
// CHECK-ENCODING: [0x9e,0x62,0x5f,0x4e]
// CHECK-ERROR: instruction requires: lut
// CHECK-UNKNOWN: 4e5f629e <unknown>

luti4   v1.8h, {v2.8h, v3.8h}, v0[0]  // 01001110-01000000-00010000-01000001
// CHECK-INST: luti4   v1.8h, { v2.8h, v3.8h }, v0[0]
// CHECK-ENCODING: [0x41,0x10,0x40,0x4e]
// CHECK-ERROR: instruction requires: lut
// CHECK-UNKNOWN: 4e401041 <unknown>

luti4   v30.8h, {v20.8h, v21.8h}, v31[3]  // 01001110-01011111-01110010-10011110
// CHECK-INST: luti4   v30.8h, { v20.8h, v21.8h }, v31[3]
// CHECK-ENCODING: [0x9e,0x72,0x5f,0x4e]
// CHECK-ERROR: instruction requires: lut
// CHECK-UNKNOWN: 4e5f729e <unknown>
