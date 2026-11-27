// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+occmo,+mte,+memtag < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+occmo,+mte,+memtag < %s \
// RUN:        | llvm-objdump -d --mattr=+occmo,+mte,+memtag --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+occmo,+mte,+memtag < %s \
// RUN:        | llvm-objdump -d --mattr=-occmo,-mte,-memtag --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+occmo,+mte,+memtag < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+occmo,+mte,+memtag -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



dc civaoc, x12
// CHECK-INST: dc civaoc, x12
// CHECK-ENCODING: encoding: [0x0c,0x7f,0x0b,0xd5]
// CHECK-ERROR: error: DC CIVAOC requires: occmo
// CHECK-UNKNOWN:  d50b7f0c      sys #3, c7, c15, #0, x12

dc cigdvaoc, x0
// CHECK-INST: dc cigdvaoc, x0
// CHECK-ENCODING: encoding: [0xe0,0x7f,0x0b,0xd5]
// CHECK-ERROR: error: DC CIGDVAOC requires: mte, memtag, occmo
// CHECK-UNKNOWN:  d50b7fe0      sys #3, c7, c15, #7, x0

dc cvaoc, x13
// CHECK-INST: dc cvaoc, x13
// CHECK-ENCODING: encoding: [0x0d,0x7b,0x0b,0xd5]
// CHECK-ERROR: error: DC CVAOC requires: occmo
// CHECK-UNKNOWN:  d50b7b0d      sys #3, c7, c11, #0, x13

dc cgdvaoc, x1
// CHECK-INST: dc cgdvaoc, x1
// CHECK-ENCODING: encoding: [0xe1,0x7b,0x0b,0xd5]
// CHECK-ERROR: error: DC CGDVAOC requires: mte, memtag, occmo
// CHECK-UNKNOWN:  d50b7be1      sys #3, c7, c11, #7, x1
