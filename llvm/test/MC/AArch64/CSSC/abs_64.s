// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v9.4a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+v9.4a < %s \
// RUN:        | llvm-objdump -d --mattr=+v9.4a - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v9.4a < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+v9.4a -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

abs     x0, x0
// CHECK-INST: abs     x0, x0
// CHECK-ENCODING: [0x00,0x20,0xc0,0xda]
// CHECK-ERROR: instruction requires: cssc

abs     x21, x10
// CHECK-INST: abs     x21, x10
// CHECK-ENCODING: [0x55,0x21,0xc0,0xda]
// CHECK-ERROR: instruction requires: cssc

abs     x23, x13
// CHECK-INST: abs     x23, x13
// CHECK-ENCODING: [0xb7,0x21,0xc0,0xda]
// CHECK-ERROR: instruction requires: cssc

abs     xzr, xzr
// CHECK-INST: abs     xzr, xzr
// CHECK-ENCODING: [0xff,0x23,0xc0,0xda]
// CHECK-ERROR: instruction requires: cssc
