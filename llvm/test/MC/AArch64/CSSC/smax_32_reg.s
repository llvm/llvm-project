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

smax    w0, w0, w0
// CHECK-INST: smax    w0, w0, w0
// CHECK-ENCODING: [0x00,0x60,0xc0,0x1a]
// CHECK-ERROR: instruction requires: cssc

smax    w21, w10, w21
// CHECK-INST: smax    w21, w10, w21
// CHECK-ENCODING: [0x55,0x61,0xd5,0x1a]
// CHECK-ERROR: instruction requires: cssc

smax    w23, w13, w8
// CHECK-INST: smax    w23, w13, w8
// CHECK-ENCODING: [0xb7,0x61,0xc8,0x1a]
// CHECK-ERROR: instruction requires: cssc

smax    wzr, wzr, wzr
// CHECK-INST: smax    wzr, wzr, wzr
// CHECK-ENCODING: [0xff,0x63,0xdf,0x1a]
// CHECK-ERROR: instruction requires: cssc
