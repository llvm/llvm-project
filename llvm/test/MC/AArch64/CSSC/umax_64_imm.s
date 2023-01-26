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

umax    x0, x0, #0
// CHECK-INST: umax    x0, x0, #0
// CHECK-ENCODING: [0x00,0x00,0xc4,0x91]
// CHECK-ERROR: instruction requires: cssc

umax    x21, x10, #85
// CHECK-INST: umax    x21, x10, #85
// CHECK-ENCODING: [0x55,0x55,0xc5,0x91]
// CHECK-ERROR: instruction requires: cssc

umax    x23, x13, #59
// CHECK-INST: umax    x23, x13, #59
// CHECK-ENCODING: [0xb7,0xed,0xc4,0x91]
// CHECK-ERROR: instruction requires: cssc

umax    xzr, xzr, #255
// CHECK-INST: umax    xzr, xzr, #255
// CHECK-ENCODING: [0xff,0xff,0xc7,0x91]
// CHECK-ERROR: instruction requires: cssc
