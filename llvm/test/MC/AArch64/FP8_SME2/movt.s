// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-lutv2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-lutv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-lutv2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST

// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-lutv2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-lutv2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme2p1,+sme-lutv2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme2p1,+sme-lutv2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movt    zt0, z0  // 11000000-01001111-00000011-11100000
// CHECK-INST: movt    zt0, z0
// CHECK-ENCODING: [0xe0,0x03,0x4f,0xc0]
// CHECK-ERROR: instruction requires: sme2 sme-lutv2
// CHECK-UNKNOWN: c04f03e0 <unknown>

movt    zt0[3, mul vl], z31  // 11000000-01001111-00110011-11111111
// CHECK-INST: movt    zt0[3, mul vl], z31
// CHECK-ENCODING: [0xff,0x33,0x4f,0xc0]
// CHECK-ERROR: instruction requires: sme2 sme-lutv2
// CHECK-UNKNOWN: c04f33ff <unknown>
