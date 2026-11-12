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

cnt     w0, w0
// CHECK-INST: cnt     w0, w0
// CHECK-ENCODING: [0x00,0x1c,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc

cnt     w21, w10
// CHECK-INST: cnt     w21, w10
// CHECK-ENCODING: [0x55,0x1d,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc

cnt     w23, w13
// CHECK-INST: cnt     w23, w13
// CHECK-ENCODING: [0xb7,0x1d,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc

cnt     wzr, wzr
// CHECK-INST: cnt     wzr, wzr
// CHECK-ENCODING: [0xff,0x1f,0xc0,0x5a]
// CHECK-ERROR: instruction requires: cssc
