// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+sve < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-sve < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG

//------------------------------------------------------------------------------
// Condition code aliases for SVE
//------------------------------------------------------------------------------

        b.none lbl
// CHECK: b.eq lbl     // encoding: [0bAAA00000,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.none lbl

        b.any lbl
// CHECK: b.ne lbl     // encoding: [0bAAA00001,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.any lbl

        b.nlast lbl
// CHECK: b.hs lbl     // encoding: [0bAAA00010,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.nlast lbl

        b.last lbl
// CHECK: b.lo lbl     // encoding: [0bAAA00011,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.last lbl

        b.first lbl
// CHECK: b.mi lbl     // encoding: [0bAAA00100,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.first lbl

        b.nfrst lbl
// CHECK: b.pl lbl     // encoding: [0bAAA00101,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.nfrst lbl

        b.pmore lbl
// CHECK: b.hi lbl     // encoding: [0bAAA01000,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.pmore lbl

        b.plast lbl
// CHECK: b.ls lbl     // encoding: [0bAAA01001,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.plast lbl

        b.tcont lbl
// CHECK: b.ge lbl     // encoding: [0bAAA01010,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.tcont lbl

        b.tstop lbl
// CHECK: b.lt lbl     // encoding: [0bAAA01011,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
// CHECK-DIAG:  invalid condition code
// CHECK-DIAG-NEXT:  b.tstop lbl
