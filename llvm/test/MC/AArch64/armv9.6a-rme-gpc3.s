// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


  apas x0
// CHECK-INST: apas x0
// CHECK-ENCODING: encoding: [0x00,0x70,0x0e,0xd5]
// CHECK-UNKNOWN:  d50e7000      apas x0

  apas x1
// CHECK-INST: apas x1
// CHECK-ENCODING: encoding: [0x01,0x70,0x0e,0xd5]
// CHECK-UNKNOWN:  d50e7001      apas x1

  apas x2
// CHECK-INST: apas x2
// CHECK-ENCODING: encoding: [0x02,0x70,0x0e,0xd5]
// CHECK-UNKNOWN:  d50e7002      apas x2

  apas x17
// CHECK-INST: apas x17
// CHECK-ENCODING: encoding: [0x11,0x70,0x0e,0xd5]
// CHECK-UNKNOWN:  d50e7011      apas x17

  apas x30
// CHECK-INST: apas x30
// CHECK-ENCODING: encoding: [0x1e,0x70,0x0e,0xd5]
// CHECK-UNKNOWN:  d50e701e      apas x30

  mrs x3, GPCBW_EL3
// CHECK-INST: mrs x3, GPCBW_EL3
// CHECK-ENCODING: encoding: [0xa3,0x21,0x3e,0xd5]
// CHECK-UNKNOWN:  d53e21a3      mrs x3, GPCBW_EL3

  msr GPCBW_EL3, x4
// CHECK-INST: msr GPCBW_EL3, x4
// CHECK-ENCODING: encoding: [0xa4,0x21,0x1e,0xd5]
// CHECK-UNKNOWN:  d51e21a4      msr GPCBW_EL3, x4
