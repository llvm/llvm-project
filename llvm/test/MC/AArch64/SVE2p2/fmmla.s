// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2,+f16mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2,+f16mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2,+f16mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2,+f16mm < %s \
// RUN:        | llvm-objdump -d --mattr=-sve2p2,-f16mm --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2,+f16mm < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2,+f16mm -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmmla z0.h, z0.h, z0.h
// CHECK-INST: fmmla z0.h, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0xa0,0x64]
// CHECK-ERROR: instruction requires: f16mm sve2p2
// CHECK-UNKNOWN: 64a0e000 <unknown>

fmmla z10.h, z10.h, z10.h
// CHECK-INST: fmmla z10.h, z10.h, z10.h
// CHECK-ENCODING: encoding: [0x4a,0xe1,0xaa,0x64]
// CHECK-ERROR: instruction requires: f16mm sve2p2
// CHECK-UNKNOWN: 64aae14a <unknown>

fmmla z21.h, z21.h, z21.h
// CHECK-INST: fmmla z21.h, z21.h, z21.h
// CHECK-ENCODING: encoding: [0xb5,0xe2,0xb5,0x64]
// CHECK-ERROR: instruction requires: f16mm sve2p2
// CHECK-UNKNOWN: 64b5e2b5 <unknown>

fmmla z31.h, z31.h, z31.h
// CHECK-INST: fmmla z31.h, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xe3,0xbf,0x64]
// CHECK-ERROR: instruction requires: f16mm sve2p2
// CHECK-UNKNOWN: 64bfe3ff <unknown>

movprfx z0, z7
fmmla z0.h, z1.h, z2.h
// CHECK-INST: movprfx z0, z7
// CHECK-INST: fmmla z0.h, z1.h, z2.h
// CHECK-ENCODING: encoding: [0x20,0xe0,0xa2,0x64]
// CHECK-ERROR: instruction requires: f16mm sve2p2
// CHECK-UNKNOWN: 64a2e020 <unknown>
