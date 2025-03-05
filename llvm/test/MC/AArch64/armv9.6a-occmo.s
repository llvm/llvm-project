// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+occmo -mattr=+mte %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding %s -mattr=+mte 2>&1 | FileCheck --check-prefix=ERROR %s
.func:
// CHECK: .func:
  dc civaoc, x12
// CHECK: dc	civaoc, x12                     // encoding: [0x0c,0x7f,0x0b,0xd5]
// ERROR: error: DC CIVAOC requires: occmo
  dc cigdvaoc, x0
// CHECK: dc	cigdvaoc, x0                    // encoding: [0xe0,0x7f,0x0b,0xd5]
// ERROR: error: DC CIGDVAOC requires: mte, memtag, occmo
  dc cvaoc, x13
// CHECK: dc	cvaoc, x13                      // encoding: [0x0d,0x7b,0x0b,0xd5]
// ERROR: error: DC CVAOC requires: occmo
  dc cgdvaoc, x1
// CHECK: dc	cgdvaoc, x1                     // encoding: [0xe1,0x7b,0x0b,0xd5]
// ERROR: error: DC CGDVAOC requires: mte, memtag, occmo

