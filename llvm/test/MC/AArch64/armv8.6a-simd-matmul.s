// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+i8mm       < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64 -show-encoding -mattr=+v8.6a      < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.6a-i8mm < %s 2>&1 | FileCheck %s --check-prefix=NOMATMUL

smmla  v1.4s, v16.16b, v31.16b
ummla  v1.4s, v16.16b, v31.16b
usmmla v1.4s, v16.16b, v31.16b
// CHECK: smmla   v1.4s, v16.16b, v31.16b // encoding: [0x01,0xa6,0x9f,0x4e]
// CHECK: ummla   v1.4s, v16.16b, v31.16b // encoding: [0x01,0xa6,0x9f,0x6e]
// CHECK: usmmla  v1.4s, v16.16b, v31.16b // encoding: [0x01,0xae,0x9f,0x4e]
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: smmla  v1.4s, v16.16b, v31.16b
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: ummla  v1.4s, v16.16b, v31.16b
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: usmmla  v1.4s, v16.16b, v31.16b

usdot v3.2s, v15.8b, v30.8b
usdot v3.4s, v15.16b, v30.16b
// CHECK: usdot   v3.2s, v15.8b, v30.8b   // encoding: [0xe3,0x9d,0x9e,0x0e]
// CHECK: usdot   v3.4s, v15.16b, v30.16b // encoding: [0xe3,0x9d,0x9e,0x4e]
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: usdot v3.2s, v15.8b, v30.8b
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: usdot v3.4s, v15.16b, v30.16b

usdot v31.2s, v1.8b,  v2.4b[3]
usdot v31.4s, v1.16b, v2.4b[3]
// CHECK: usdot   v31.2s, v1.8b, v2.4b[3] // encoding: [0x3f,0xf8,0xa2,0x0f]
// CHECK: usdot   v31.4s, v1.16b, v2.4b[3] // encoding: [0x3f,0xf8,0xa2,0x4f]
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: usdot   v31.2s, v1.8b, v2.4b[3]
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: usdot   v31.4s, v1.16b, v2.4b[3]

sudot v31.2s, v1.8b,  v2.4b[3]
sudot v31.4s, v1.16b, v2.4b[3]
// CHECK: sudot   v31.2s, v1.8b, v2.4b[3] // encoding: [0x3f,0xf8,0x22,0x0f]
// CHECK: sudot   v31.4s, v1.16b, v2.4b[3] // encoding: [0x3f,0xf8,0x22,0x4f]
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: sudot   v31.2s, v1.8b, v2.4b[3]
// NOMATMUL: instruction requires: i8mm
// NOMATMUL-NEXT: sudot   v31.4s, v1.16b, v2.4b[3]
