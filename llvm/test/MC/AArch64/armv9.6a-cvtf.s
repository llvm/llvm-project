// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fprcvt < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fprcvt < %s \
// RUN:        | llvm-objdump -d --mattr=+fprcvt - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fprcvt < %s \
// RUN:        | llvm-objdump -d  --no-print-imm-hex --mattr=-fprcvt - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fprcvt < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fprcvt -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

scvtf d1, s2
// CHECK-INST: scvtf d1, s2
// CHECK-ENCODING: [0x41,0x00,0x7c,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e7c0041 <unknown>

scvtf h1, s2
// CHECK-INST: scvtf h1, s2
// CHECK-ENCODING: [0x41,0x00,0xfc,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1efc0041 <unknown>

scvtf h2, d0
// CHECK-INST: scvtf h2, d0
// CHECK-ENCODING: [0x02,0x00,0xfc,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9efc0002 <unknown>

scvtf s3, d4
// CHECK-INST: scvtf s3, d4
// CHECK-ENCODING: [0x83,0x00,0x3c,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3c0083 <unknown>

ucvtf d1, s2
// CHECK-INST: ucvtf d1, s2
// CHECK-ENCODING: [0x41,0x00,0x7d,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1e7d0041 <unknown>

ucvtf h1, s2
// CHECK-INST: ucvtf h1, s2
// CHECK-ENCODING: [0x41,0x00,0xfd,0x1e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 1efd0041 <unknown>

ucvtf h2, d0
// CHECK-INST: ucvtf h2, d0
// CHECK-ENCODING: [0x02,0x00,0xfd,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9efd0002 <unknown>

ucvtf s3, d4
// CHECK-INST: ucvtf s3, d4
// CHECK-ENCODING: [0x83,0x00,0x3d,0x9e]
// CHECK-ERROR: instruction requires: fprcvt
// CHECK-UNKNOWN: 9e3d0083 <unknown>