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

abs     w0, w0
// CHECK-INST: abs     w0, w0
// CHECK-ENCODING: [0x00,0x20,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc

abs     w21, w10
// CHECK-INST: abs     w21, w10
// CHECK-ENCODING: [0x55,0x21,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc

abs     w23, w13
// CHECK-INST: abs     w23, w13
// CHECK-ENCODING: [0xb7,0x21,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc

abs     wzr, wzr
// CHECK-INST: abs     wzr, wzr
// CHECK-ENCODING: [0xff,0x23,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc
