// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-b16mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve-b16mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve-b16mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve-b16mm < %s \
// RUN:        | llvm-objdump -d --mattr=-sve-b16mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-b16mm < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve-b16mm -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

bfmmla z0.h, z0.h, z0.h
// CHECK-INST: bfmmla z0.h, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0xe0,0x64]
// CHECK-ERROR: instruction requires: sve-b16mm
// CHECK-UNKNOWN: 64e0e000 <unknown>

bfmmla z10.h, z10.h, z10.h
// CHECK-INST: bfmmla z10.h, z10.h, z10.h
// CHECK-ENCODING: encoding: [0x4a,0xe1,0xea,0x64]
// CHECK-ERROR: instruction requires: sve-b16mm
// CHECK-UNKNOWN: 64eae14a <unknown>

bfmmla z21.h, z21.h, z21.h
// CHECK-INST: bfmmla z21.h, z21.h, z21.h
// CHECK-ENCODING: encoding: [0xb5,0xe2,0xf5,0x64]
// CHECK-ERROR: instruction requires: sve-b16mm
// CHECK-UNKNOWN: 64f5e2b5 <unknown>

bfmmla z31.h, z31.h, z31.h
// CHECK-INST: bfmmla z31.h, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xe3,0xff,0x64]
// CHECK-ERROR: instruction requires: sve-b16mm
// CHECK-UNKNOWN: 64ffe3ff <unknown>

movprfx z0, z7
bfmmla z0.h, z1.h, z2.h
// CHECK-INST: movprfx z0, z7
// CHECK-INST: bfmmla z0.h, z1.h, z2.h
// CHECK-ENCODING: encoding: [0x20,0xe0,0xe2,0x64]
// CHECK-ERROR: instruction requires: sve-b16mm
// CHECK-UNKNOWN: 64e2e020 <unknown>
