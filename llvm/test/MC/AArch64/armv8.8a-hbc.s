// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+hbc < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.8a < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2>&1 | FileCheck --check-prefix=CHECK-NO-HBC-ERR %s

lbl:
        bc.eq lbl
        bc.ne lbl
        bc.cs lbl
        bc.hs lbl
        bc.lo lbl
        bc.cc lbl
        bc.mi lbl
        bc.pl lbl
        bc.vs lbl
        bc.vc lbl
        bc.hi lbl
        bc.ls lbl
        bc.ge lbl
        bc.lt lbl
        bc.gt lbl
        bc.le lbl
        bc.al lbl

// CHECK: bc.eq lbl                    // encoding: [0bAAA10000,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.ne lbl                    // encoding: [0bAAA10001,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.hs lbl                    // encoding: [0bAAA10010,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.hs lbl                    // encoding: [0bAAA10010,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.lo lbl                    // encoding: [0bAAA10011,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.lo lbl                    // encoding: [0bAAA10011,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.mi lbl                    // encoding: [0bAAA10100,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.pl lbl                    // encoding: [0bAAA10101,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.vs lbl                    // encoding: [0bAAA10110,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.vc lbl                    // encoding: [0bAAA10111,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.hi lbl                    // encoding: [0bAAA11000,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.ls lbl                    // encoding: [0bAAA11001,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.ge lbl                    // encoding: [0bAAA11010,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.lt lbl                    // encoding: [0bAAA11011,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.gt lbl                    // encoding: [0bAAA11100,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.le lbl                    // encoding: [0bAAA11101,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK: bc.al lbl                    // encoding: [0bAAA11110,A,A,0x54]
// CHECK:                              //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
// CHECK-NO-HBC-ERR: [[@LINE-53]]:9: error: instruction requires: hbc
