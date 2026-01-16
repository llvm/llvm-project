// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+tev < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+tev < %s \
// RUN:        | llvm-objdump -d --mattr=+tev --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+tev < %s \
// RUN:        | llvm-objdump -d --mattr=-tev --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+tev < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+tev -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// TIndex Exception-like Vector (FEAT_TEV).
//------------------------------------------------------------------------------

TENTER #32
// CHECK-INST:    tenter #32
// CHECK-ENCODING: [0x00,0x04,0xe0,0xd4]
// CHECK-UNKNOWN: d4e00400
// CHECK-ERROR: error: instruction requires: tev

TENTER #32, NB
// CHECK-INST:    tenter #32, nb
// CHECK-ENCODING: [0x00,0x04,0xe2,0xd4]
// CHECK-UNKNOWN: d4e20400
// CHECK-ERROR: error: instruction requires: tev

TEXIT
// CHECK-INST:    texit
// CHECK-ENCODING: [0xe0,0x03,0xff,0xd6]
// CHECK-UNKNOWN: d6ff03e0
// CHECK-ERROR: error: instruction requires: tev

TEXIT NB
// CHECK-INST:    texit nb
// CHECK-ENCODING: [0xe0,0x07,0xff,0xd6]
// CHECK-UNKNOWN: d6ff07e0
// CHECK-ERROR: error: instruction requires: tev
