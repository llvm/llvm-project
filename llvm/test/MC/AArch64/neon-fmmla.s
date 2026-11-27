// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f16mm< %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f16mm < %s \
// RUN:        | llvm-objdump -d --mattr=+f16mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+f16mm < %s \
// RUN:        | llvm-objdump -d --mattr=-f16mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+f16mm < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+f16mm -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmmla v0.8h, v0.8h, v0.8h
// CHECK-INST: fmmla v0.8h, v0.8h, v0.8h
// CHECK-ENCODING: encoding: [0x00,0xec,0xc0,0x4e]
// CHECK-ERROR: instruction requires: f16mm
// CHECK-UNKNOWN: 4ec0ec00 <unknown>

fmmla v10.8h, v10.8h, v10.8h
// CHECK-INST: fmmla v10.8h, v10.8h, v10.8h
// CHECK-ENCODING: encoding: [0x4a,0xed,0xca,0x4e]
// CHECK-ERROR: instruction requires: f16mm
// CHECK-UNKNOWN: 4ecaed4a <unknown>

fmmla v21.8h, v21.8h, v21.8h
// CHECK-INST: fmmla v21.8h, v21.8h, v21.8h
// CHECK-ENCODING: encoding: [0xb5,0xee,0xd5,0x4e]
// CHECK-ERROR: instruction requires: f16mm
// CHECK-UNKNOWN: 4ed5eeb5 <unknown>

fmmla v31.8h, v31.8h, v31.8h
// CHECK-INST: fmmla v31.8h, v31.8h, v31.8h
// CHECK-ENCODING: encoding: [0xff,0xef,0xdf,0x4e]
// CHECK-ERROR: instruction requires: f16mm
// CHECK-UNKNOWN: 4edfefff <unknown>
