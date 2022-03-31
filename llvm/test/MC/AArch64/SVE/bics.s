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

bics    p0.b, p0/z, p0.b, p0.b
// CHECK-INST: bics    p0.b, p0/z, p0.b, p0.b
// CHECK-ENCODING: [0x10,0x40,0x40,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 10 40 40 25 <unknown>

bics    p15.b, p15/z, p15.b, p15.b
// CHECK-INST: bics    p15.b, p15/z, p15.b, p15.b
// CHECK-ENCODING: [0xff,0x7d,0x4f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 7d 4f 25 <unknown>
