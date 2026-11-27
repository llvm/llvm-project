// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v8.9a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v9.4a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+specres2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+specres2 < %s \
// RUN:        | llvm-objdump -d --mattr=+specres2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+specres2 < %s \
// RUN:   | llvm-objdump -d --mattr=-specres2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+specres2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+specres2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



cosp rctx, x0
// CHECK-INST: cosp rctx, x0
// CHECK-ENCODING: encoding: [0xc0,0x73,0x0b,0xd5]
// CHECK-ERROR: error: COSP requires: predres2
// CHECK-UNKNOWN:  d50b73c0      sys #3, c7, c3, #6, x0

sys #3, c7, c3, #6, x0
// CHECK-INST: cosp rctx, x0
// CHECK-ENCODING: encoding: [0xc0,0x73,0x0b,0xd5]
// CHECK-UNKNOWN:  d50b73c0      sys #3, c7, c3, #6, x0


