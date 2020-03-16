// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

brkpb  p0.b,  p15/z, p1.b,  p2.b
// CHECK-INST: brkpb	p0.b, p15/z, p1.b, p2.b
// CHECK-ENCODING: [0x30,0xfc,0x02,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 30 fc 02 25 <unknown>

brkpb  p15.b, p15/z, p15.b, p15.b
// CHECK-INST: brkpb	p15.b, p15/z, p15.b, p15.b
// CHECK-ENCODING: [0xff,0xfd,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff fd 0f 25 <unknown>
