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

umin    x0, x0, x0
// CHECK-INST: umin    x0, x0, x0
// CHECK-ENCODING: [0x00,0x6c,0xc0,0x9a]
// CHECK-ERROR: instruction requires: cssc

umin    x21, x10, x21
// CHECK-INST: umin    x21, x10, x21
// CHECK-ENCODING: [0x55,0x6d,0xd5,0x9a]
// CHECK-ERROR: instruction requires: cssc

umin    x23, x13, x8
// CHECK-INST: umin    x23, x13, x8
// CHECK-ENCODING: [0xb7,0x6d,0xc8,0x9a]
// CHECK-ERROR: instruction requires: cssc

umin    xzr, xzr, xzr
// CHECK-INST: umin    xzr, xzr, xzr
// CHECK-ENCODING: [0xff,0x6f,0xdf,0x9a]
// CHECK-ERROR: instruction requires: cssc
