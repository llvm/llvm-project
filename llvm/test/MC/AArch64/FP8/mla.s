// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8fma < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fp8fma < %s \
// RUN:        | llvm-objdump -d --mattr=+fp8fma - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fp8fma < %s \
// RUN:        | llvm-objdump -d --mattr=-fp8fma - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8fma < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fp8fma -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

/// VECTOR
// MLA
fmlalb  v0.8h, v0.16b, v0.16b
// CHECK-INST: fmlalb  v0.8h, v0.16b, v0.16b
// CHECK-ENCODING: [0x00,0xfc,0xc0,0x0e]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 0ec0fc00 <unknown>

fmlalt  v31.8h, v31.16b, v31.16b
// CHECK-INST: fmlalt  v31.8h, v31.16b, v31.16b
// CHECK-ENCODING: [0xff,0xff,0xdf,0x4e]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 4edfffff <unknown>


// MLALL
fmlallbb  v0.4s, v0.16b, v31.16b
// CHECK-INST: fmlallbb  v0.4s, v0.16b, v31.16b
// CHECK-ENCODING: [0x00,0xc4,0x1f,0x0e]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 0e1fc400 <unknown>

fmlallbt  v31.4s, v31.16b, v0.16b
// CHECK-INST: fmlallbt  v31.4s, v31.16b, v0.16b
// CHECK-ENCODING: [0xff,0xc7,0x40,0x0e]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 0e40c7ff <unknown>

fmlalltb  v31.4s, v31.16b, v0.16b
// CHECK-INST: fmlalltb  v31.4s, v31.16b, v0.16b
// CHECK-ENCODING: [0xff,0xc7,0x00,0x4e]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 4e00c7ff <unknown>

fmlalltt  v0.4s, v0.16b, v31.16b
// CHECK-INST: fmlalltt  v0.4s, v0.16b, v31.16b
// CHECK-ENCODING: [0x00,0xc4,0x5f,0x4e]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 4e5fc400 <unknown>

fmlalltt v31.4s, v31.16b, v31.16b  // 01001110-01011111-11000111-11111111
// CHECK-INST: fmlalltt v31.4s, v31.16b, v31.16b
// CHECK-ENCODING: [0xff,0xc7,0x5f,0x4e]
// CHECK-ERROR: instruction requires: fp8
// CHECK-UNKNOWN: 4e5fc7ff <unknown>

//INDEXED
// MLA
fmlalb  v31.8h, v0.16b, v0.b[0]
// CHECK-INST: fmlalb  v31.8h, v0.16b, v0.b[0]
// CHECK-ENCODING: [0x1f,0x00,0xc0,0x0f]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 0fc0001f <unknown>

fmlalt  v31.8h, v0.16b, v0.b[15]
// CHECK-INST: fmlalt  v31.8h, v0.16b, v0.b[15]
// CHECK-ENCODING: [0x1f,0x08,0xf8,0x4f]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 4ff8081f <unknown>

// MLALL
fmlallbb  v31.4s, v0.16b, v7.b[0]
// CHECK-INST: fmlallbb  v31.4s, v0.16b, v7.b[0]
// CHECK-ENCODING: [0x1f,0x80,0x07,0x2f]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 2f07801f <unknown>

fmlalltt  v31.4s, v0.16b, v7.b[0]
// CHECK-INST: fmlalltt  v31.4s, v0.16b, v7.b[0]
// CHECK-ENCODING: [0x1f,0x80,0x47,0x6f]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 6f47801f <unknown>

fmlalltb  v0.4s, v31.16b, v7.b[15]
// CHECK-INST: fmlalltb  v0.4s, v31.16b, v7.b[15]
// CHECK-ENCODING: [0xe0,0x8b,0x3f,0x6f]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 6f3f8be0 <unknown>

fmlallbt  v0.4s, v31.16b, v0.b[15]
// CHECK-INST: fmlallbt  v0.4s, v31.16b, v0.b[15]
// CHECK-ENCODING: [0xe0,0x8b,0x78,0x2f]
// CHECK-ERROR: instruction requires: fp8fma
// CHECK-UNKNOWN: 2f788be0 <unknown>
