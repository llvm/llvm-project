// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f8f16mm,+f8f32mm  < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f8f16mm,+f8f32mm  < %s \
// RUN:        | llvm-objdump -d --mattr=+f8f16mm,+f8f32mm  - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f8f16mm,+f8f32mm  < %s \
// RUN:        | llvm-objdump -d  --no-print-imm-hex --mattr=-f8f16mm,-f8f32mm - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f8f16mm,+f8f32mm  < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+f8f16mm,+f8f32mm  -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmmla v0.8h, v1.16b, v2.16b
// CHECK-INST: fmmla v0.8h, v1.16b, v2.16b
// CHECK-ENCODING: [0x20,0xec,0x02,0x6e]
// CHECK-ERROR: instruction requires: f8f16mm
// CHECK-UNKNOWN: 6e02ec20 <unknown>

fmmla v0.4s, v1.16b, v2.16b
// CHECK-INST: fmmla v0.4s, v1.16b, v2.16b
// CHECK-ENCODING: [0x20,0xec,0x82,0x6e]
// CHECK-ERROR: instruction requires: f8f32mm
// CHECK-UNKNOWN: 6e82ec20 <unknown>