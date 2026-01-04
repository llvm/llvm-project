// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f16f32dot < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f16f32dot < %s \
// RUN:        | llvm-objdump -d --mattr=+f16f32dot --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f16f32dot < %s \
// RUN:        | llvm-objdump -d --mattr=-f16f32dot --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f16f32dot < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+f16f32dot -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fdot v0.2s, v0.4h, v0.4h
// CHECK-INST: fdot v0.2s, v0.4h, v0.4h
// CHECK-ENCODING: encoding: [0x00,0xfc,0x80,0x0e]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0e80fc00 <unknown>

fdot v10.2s, v10.4h, v10.4h
// CHECK-INST: fdot v10.2s, v10.4h, v10.4h
// CHECK-ENCODING: encoding: [0x4a,0xfd,0x8a,0x0e]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0e8afd4a <unknown>

fdot v31.2s, v31.4h, v31.4h
// CHECK-INST: fdot v31.2s, v31.4h, v31.4h
// CHECK-ENCODING: encoding: [0xff,0xff,0x9f,0x0e]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0e9fffff <unknown>

fdot v0.4s, v0.8h, v0.8h
// CHECK-INST: fdot v0.4s, v0.8h, v0.8h
// CHECK-ENCODING: encoding: [0x00,0xfc,0x80,0x4e]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4e80fc00 <unknown>

fdot v10.4s, v10.8h, v10.8h
// CHECK-INST: fdot v10.4s, v10.8h, v10.8h
// CHECK-ENCODING: encoding: [0x4a,0xfd,0x8a,0x4e]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4e8afd4a <unknown>

fdot v31.4s, v31.8h, v31.8h
// CHECK-INST: fdot v31.4s, v31.8h, v31.8h
// CHECK-ENCODING: encoding: [0xff,0xff,0x9f,0x4e]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4e9fffff <unknown>

// fdot indexed

fdot v0.2s, v0.4h, v0.2h[0]
// CHECK-INST: fdot v0.2s, v0.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x00,0x90,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f409000 <unknown>

fdot v10.2s, v0.4h, v0.2h[0]
// CHECK-INST: fdot v10.2s, v0.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x0a,0x90,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f40900a <unknown>

fdot v21.2s, v0.4h, v0.2h[0]
// CHECK-INST: fdot v21.2s, v0.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x15,0x90,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f409015 <unknown>

fdot v31.2s, v0.4h, v0.2h[0]
// CHECK-INST: fdot v31.2s, v0.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x1f,0x90,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f40901f <unknown>

fdot v0.2s, v10.4h, v0.2h[0]
// CHECK-INST: fdot v0.2s, v10.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x40,0x91,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f409140 <unknown>

fdot v10.2s, v10.4h, v0.2h[0]
// CHECK-INST: fdot v10.2s, v10.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x4a,0x91,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f40914a <unknown>

fdot v21.2s, v10.4h, v0.2h[0]
// CHECK-INST: fdot v21.2s, v10.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x55,0x91,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f409155 <unknown>

fdot v31.2s, v10.4h, v0.2h[0]
// CHECK-INST: fdot v31.2s, v10.4h, v0.2h[0]
// CHECK-ENCODING: encoding: [0x5f,0x91,0x40,0x0f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 0f40915f <unknown>

fdot v0.4s, v21.8h, v31.2h[3]
// CHECK-INST: fdot v0.4s, v21.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xa0,0x9a,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9aa0 <unknown>

fdot v10.4s, v21.8h, v31.2h[3]
// CHECK-INST: fdot v10.4s, v21.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xaa,0x9a,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9aaa <unknown>

fdot v21.4s, v21.8h, v31.2h[3]
// CHECK-INST: fdot v21.4s, v21.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xb5,0x9a,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9ab5 <unknown>

fdot v31.4s, v21.8h, v31.2h[3]
// CHECK-INST: fdot v31.4s, v21.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xbf,0x9a,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9abf <unknown>

fdot v0.4s, v31.8h, v31.2h[3]
// CHECK-INST: fdot v0.4s, v31.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xe0,0x9b,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9be0 <unknown>

fdot v10.4s, v31.8h, v31.2h[3]
// CHECK-INST: fdot v10.4s, v31.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xea,0x9b,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9bea <unknown>

fdot v21.4s, v31.8h, v31.2h[3]
// CHECK-INST: fdot v21.4s, v31.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xf5,0x9b,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9bf5 <unknown>

fdot v31.4s, v31.8h, v31.2h[3]
// CHECK-INST: fdot v31.4s, v31.8h, v31.2h[3]
// CHECK-ENCODING: encoding: [0xff,0x9b,0x7f,0x4f]
// CHECK-ERROR: instruction requires: f16f32dot
// CHECK-UNKNOWN: 4f7f9bff <unknown>
