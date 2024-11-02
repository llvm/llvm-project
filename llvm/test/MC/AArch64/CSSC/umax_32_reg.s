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

umax    w0, w0, w0
// CHECK-INST: umax    w0, w0, w0
// CHECK-ENCODING: [0x00,0x64,0xc0,0x1a]
// CHECK-ERROR: instruction requires: cssc

umax    w21, w10, w21
// CHECK-INST: umax    w21, w10, w21
// CHECK-ENCODING: [0x55,0x65,0xd5,0x1a]
// CHECK-ERROR: instruction requires: cssc

umax    w23, w13, w8
// CHECK-INST: umax    w23, w13, w8
// CHECK-ENCODING: [0xb7,0x65,0xc8,0x1a]
// CHECK-ERROR: instruction requires: cssc

umax    wzr, wzr, wzr
// CHECK-INST: umax    wzr, wzr, wzr
// CHECK-ENCODING: [0xff,0x67,0xdf,0x1a]
// CHECK-ERROR: instruction requires: cssc
