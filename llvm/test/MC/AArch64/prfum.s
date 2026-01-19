// RUN: llvm-mc -triple=aarch64 -show-encoding --print-imm-hex=false < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --print-imm-hex=false - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding --print-imm-hex=false \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// PRFM falls back to PRFUM for negative or unaligned offsets (not a multiple
// of 8).

prfm pldl1keep, [x0, #-256]
// CHECK-INST: prfum pldl1keep, [x0, #-256]
// CHECK-ENCODING: [0x00,0x00,0x90,0xf8]

prfm pldl1keep, [x0, #-8]
// CHECK-INST: prfum pldl1keep, [x0, #-8]
// CHECK-ENCODING: [0x00,0x80,0x9f,0xf8]

prfm pldl1keep, [x0, #-1]
// CHECK-INST: prfum pldl1keep, [x0, #-1]
// CHECK-ENCODING: [0x00,0xf0,0x9f,0xf8]

prfm pldl1keep, [x0, #0]
// CHECK-INST: prfm pldl1keep, [x0]
// CHECK-ENCODING: [0x00,0x00,0x80,0xf9]

prfm pldl1keep, [x0, #1]
// CHECK-INST: prfum pldl1keep, [x0, #1]
// CHECK-ENCODING: [0x00,0x10,0x80,0xf8]

prfm pldl1keep, [x0, #8]
// CHECK-INST: prfm pldl1keep, [x0, #8]
// CHECK-ENCODING: [0x00,0x04,0x80,0xf9]

prfm pldl1keep, [x0, #255]
// CHECK-INST: prfum pldl1keep, [x0, #255]
// CHECK-ENCODING: [0x00,0xf0,0x8f,0xf8]

prfm pldl1keep, [x0, #256]
// CHECK-INST: prfm pldl1keep, [x0, #256]
// CHECK-ENCODING: [0x00,0x80,0x80,0xf9]
