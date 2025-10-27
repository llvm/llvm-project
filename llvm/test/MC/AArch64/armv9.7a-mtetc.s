// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mtetc < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mtetc < %s \
// RUN:        | llvm-objdump -d --mattr=+mtetc --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mtetc < %s \
// RUN:        | llvm-objdump -d --mattr=-mtetc --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mtetc < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+mtetc -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// FEAT_MTETC Extension instructions
//------------------------------------------------------------------------------

dc zgbva, x0
// CHECK-INST:    dc zgbva, x0
// CHECK-ENCODING: [0xa0,0x74,0x0b,0xd5]
// CHECK-UNKNOWN: d50b74a0 sys #3, c7, c4, #5, x0
// CHECK-ERROR: DC ZGBVA requires: mtetc

dc gbva, x0
// CHECK-INST:    dc gbva, x0
// CHECK-ENCODING: [0xe0,0x74,0x0b,0xd5]
// CHECK-UNKNOWN: d50b74e0 sys #3, c7, c4, #7, x0
// CHECK-ERROR: DC GBVA requires: mtetc
