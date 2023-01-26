// RUN: llvm-mc -triple=aarch64 -show-encoding --print-imm-hex=false -mattr=+v9.4a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+v9.4a < %s \
// RUN:        | llvm-objdump -d --print-imm-hex=false --mattr=+v9.4a - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v9.4a < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+v9.4a -disassemble -show-encoding \
// RUN:           --print-imm-hex=false \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

umin    x0, x0, #0
// CHECK-INST: umin    x0, x0, #0
// CHECK-ENCODING: [0x00,0x00,0xcc,0x91]
// CHECK-ERROR: instruction requires: cssc

umin    x21, x10, #85
// CHECK-INST: umin    x21, x10, #85
// CHECK-ENCODING: [0x55,0x55,0xcd,0x91]
// CHECK-ERROR: instruction requires: cssc

umin    x23, x13, #59
// CHECK-INST: umin    x23, x13, #59
// CHECK-ENCODING: [0xb7,0xed,0xcc,0x91]
// CHECK-ERROR: instruction requires: cssc

umin    xzr, xzr, #255
// CHECK-INST: umin    xzr, xzr, #255
// CHECK-ENCODING: [0xff,0xff,0xcf,0x91]
// CHECK-ERROR: instruction requires: cssc
