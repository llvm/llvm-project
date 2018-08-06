// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

rdvl    x0, #0
// CHECK-INST: rdvl    x0, #0
// CHECK-ENCODING: [0x00,0x50,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 50 bf 04 <unknown>

rdvl    xzr, #-1
// CHECK-INST: rdvl    xzr, #-1
// CHECK-ENCODING: [0xff,0x57,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 57 bf 04 <unknown>

rdvl    x23, #31
// CHECK-INST: rdvl    x23, #31
// CHECK-ENCODING: [0xf7,0x53,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: f7 53 bf 04 <unknown>

rdvl    x21, #-32
// CHECK-INST: rdvl    x21, #-32
// CHECK-ENCODING: [0x15,0x54,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 15 54 bf 04 <unknown>
