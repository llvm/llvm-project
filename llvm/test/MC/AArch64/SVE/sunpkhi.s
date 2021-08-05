// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sunpkhi z31.h, z31.b
// CHECK-INST: sunpkhi	z31.h, z31.b
// CHECK-ENCODING: [0xff,0x3b,0x71,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b 71 05 <unknown>

sunpkhi z31.s, z31.h
// CHECK-INST: sunpkhi	z31.s, z31.h
// CHECK-ENCODING: [0xff,0x3b,0xb1,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b b1 05 <unknown>

sunpkhi z31.d, z31.s
// CHECK-INST: sunpkhi	z31.d, z31.s
// CHECK-ENCODING: [0xff,0x3b,0xf1,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 3b f1 05 <unknown>
