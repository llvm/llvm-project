// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+tlbid,+poe2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+poe2 < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+tlbid,+poe2 < %s \
// RUN:        | llvm-objdump -d --mattr=+tlbid,+poe2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+tlbid,+poe2 < %s \
// RUN:        | llvm-objdump -d --mattr=-tlbid,-poe2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+tlbid,+poe2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+tlbid,+poe2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// FEAT_TLBID and POE2 combined

plbi alle2is, x0
// CHECK-INST: plbi alle2is, x0
// CHECK-ENCODING: encoding: [0x00,0xa3,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50ca300 sys #4, c10, c3, #0, x0

plbi alle2os, x0
// CHECK-INST: plbi alle2os, x0
// CHECK-ENCODING: encoding: [0x00,0xa1,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50ca100 sys #4, c10, c1, #0, x0

plbi alle1is, x0
// CHECK-INST: plbi alle1is, x0
// CHECK-ENCODING: encoding: [0x80,0xa3,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50ca380 sys #4, c10, c3, #4, x0

plbi alle1os, x0
// CHECK-INST: plbi alle1os, x0
// CHECK-ENCODING: encoding: [0x80,0xa1,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50ca180 sys #4, c10, c1, #4, x0

plbi vmalle1is, x0
// CHECK-INST: plbi vmalle1is, x0
// CHECK-ENCODING: encoding: [0x00,0xa3,0x08,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d508a300 sys #0, c10, c3, #0, x0

plbi vmalle1os, x0
// CHECK-INST: plbi vmalle1os, x0
// CHECK-ENCODING: encoding: [0x00,0xa1,0x08,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d508a100 sys #0, c10, c1, #0, x0

plbi alle2isnxs, x0
// CHECK-INST: plbi alle2isnxs, x0
// CHECK-ENCODING: encoding: [0x00,0xab,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50cab00 sys #4, c10, c11, #0, x0

plbi alle2osnxs, x0
// CHECK-INST: plbi alle2osnxs, x0
// CHECK-ENCODING: encoding: [0x00,0xa9,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50ca900 sys #4, c10, c9, #0, x0

plbi alle1isnxs, x0
// CHECK-INST: plbi alle1isnxs, x0
// CHECK-ENCODING: encoding: [0x80,0xab,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50cab80 sys #4, c10, c11, #4, x0

plbi alle1osnxs, x0
// CHECK-INST: plbi alle1osnxs, x0
// CHECK-ENCODING: encoding: [0x80,0xa9,0x0c,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d50ca980 sys #4, c10, c9, #4, x0

plbi vmalle1isnxs, x0
// CHECK-INST: plbi vmalle1isnxs, x0
// CHECK-ENCODING: encoding: [0x00,0xab,0x08,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d508ab00 sys #0, c10, c11, #0, x0

plbi vmalle1osnxs, x0
// CHECK-INST: plbi vmalle1osnxs, x0
// CHECK-ENCODING: encoding: [0x00,0xa9,0x08,0xd5]
// CHECK-ERROR: error: specified plbi op does not use a register
// CHECK-UNKNOWN: d508a900 sys #0, c10, c9, #0, x0
