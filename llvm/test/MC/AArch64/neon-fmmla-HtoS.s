// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f16f32mm< %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f16f32mm < %s \
// RUN:        | llvm-objdump -d --mattr=+f16f32mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f16f32mm < %s \
// RUN:        | llvm-objdump -d --mattr=-f16f32mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f16f32mm < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+f16f32mm -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmmla v0.4s, v0.8h, v0.8h
// CHECK-INST: fmmla v0.4s, v0.8h, v0.8h
// CHECK-ENCODING: encoding: [0x00,0xec,0x40,0x4e]
// CHECK-ERROR: instruction requires: f16f32mm
// CHECK-UNKNOWN: 4e40ec00 <unknown>

fmmla v10.4s, v10.8h, v10.8h
// CHECK-INST: fmmla v10.4s, v10.8h, v10.8h
// CHECK-ENCODING: encoding: [0x4a,0xed,0x4a,0x4e]
// CHECK-ERROR: instruction requires: f16f32mm
// CHECK-UNKNOWN: 4e4aed4a <unknown>

fmmla v21.4s, v21.8h, v21.8h
// CHECK-INST: fmmla v21.4s, v21.8h, v21.8h
// CHECK-ENCODING: encoding: [0xb5,0xee,0x55,0x4e]
// CHECK-ERROR: instruction requires: f16f32mm
// CHECK-UNKNOWN: 4e55eeb5 <unknown>

fmmla v31.4s, v31.8h, v31.8h
// CHECK-INST: fmmla v31.4s, v31.8h, v31.8h
// CHECK-ENCODING: encoding: [0xff,0xef,0x5f,0x4e]
// CHECK-ERROR: instruction requires: f16f32mm
// CHECK-UNKNOWN: 4e5fefff <unknown>
