// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8dot2,+fp8dot4 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fp8dot2,+fp8dot4 < %s \
// RUN:        | llvm-objdump -d --mattr=+fp8dot2,+fp8dot4 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fp8dot2,+fp8dot4 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8dot2,+fp8dot4 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fp8dot2,+fp8dot4 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

/// VECTOR
fdot  v31.4h, v0.8b, v0.8b
// CHECK-INST: fdot  v31.4h, v0.8b, v0.8b
// CHECK-ENCODING: [0x1f,0xfc,0x40,0x0e]
// CHECK-ERROR: instruction requires: fp8dot2
// CHECK-UNKNOWN: 0e40fc1f  <unknown>

fdot  v31.8h, v0.16b, v31.16b
// CHECK-INST: fdot  v31.8h, v0.16b, v31.16b
// CHECK-ENCODING:  [0x1f,0xfc,0x5f,0x4e]
// CHECK-ERROR: instruction requires: fp8dot2
// CHECK-UNKNOWN: 4e5ffc1f  <unknown>

fdot  v0.2s, v0.8b, v31.8b
// CHECK-INST: fdot   v0.2s, v0.8b, v31.8b
// CHECK-ENCODING:  [0x00,0xfc,0x1f,0x0e]
// CHECK-ERROR: instruction requires: fp8dot4
// CHECK-UNKNOWN: 0e1ffc00  <unknown>

fdot  v31.4s, v0.16b, v31.16b
// CHECK-INST: fdot  v31.4s, v0.16b, v31.16b
// CHECK-ENCODING:  [0x1f,0xfc,0x1f,0x4e]
// CHECK-ERROR: instruction requires: fp8dot4
// CHECK-UNKNOWN: 4e1ffc1f  <unknown>

//INDEXED
fdot  v31.4h, v31.8b, v15.2b[0]
// CHECK-INST: fdot  v31.4h, v31.8b, v15.2b[0]
// CHECK-ENCODING: [0xff,0x03,0x4f,0x0f]
// CHECK-ERROR: instruction requires: fp8dot2
// CHECK-UNKNOWN: 0f4f03ff <unknown>

fdot v26.8H, v22.16B, v9.2B[0]
// CHECK-INST: fdot  v26.8h, v22.16b, v9.2b[0]
// CHECK-ENCODING: [0xda,0x02,0x49,0x4f]
// CHECK-ERROR: instruction requires: fp8dot2
// CHECK-UNKNOWN: 4f4902da <unknown>

fdot  v0.8h, v0.16b, v15.2b[7]
// CHECK-INST: fdot  v0.8h, v0.16b, v15.2b[7]
// CHECK-ENCODING: [0x00,0x08,0x7f,0x4f]
// CHECK-ERROR: instruction requires: fp8dot2
// CHECK-UNKNOWN: 4f7f0800 <unknown>

fdot  v0.2s, v0.8b, v31.4b[0]
// CHECK-INST: fdot  v0.2s, v0.8b, v31.4b[0]
// CHECK-ENCODING: [0x00,0x00,0x1f,0x0f]
// CHECK-ERROR: instruction requires: fp8dot4
// CHECK-UNKNOWN: 0f1f0000  <unknown>

fdot  v0.4s, v31.16b, v0.4b[3]
// CHECK-INST: fdot  v0.4s, v31.16b, v0.4b[3]
// CHECK-ENCODING: [0xe0,0x0b,0x20,0x4f]
// CHECK-ERROR: instruction requires: fp8dot4
// CHECK-UNKNOWN: 4f200be0  <unknown>
