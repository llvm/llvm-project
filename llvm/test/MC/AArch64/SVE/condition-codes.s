// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Condition code aliases for SVE.
// They don't actually require +sve since they're just aliases.
//------------------------------------------------------------------------------

        b.none lbl
// CHECK: b.eq lbl     // encoding: [0bAAA00000,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.any lbl
// CHECK: b.ne lbl     // encoding: [0bAAA00001,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.nlast lbl
// CHECK: b.hs lbl     // encoding: [0bAAA00010,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.last lbl
// CHECK: b.lo lbl     // encoding: [0bAAA00011,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.first lbl
// CHECK: b.mi lbl     // encoding: [0bAAA00100,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.nfrst lbl
// CHECK: b.pl lbl     // encoding: [0bAAA00101,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.pmore lbl
// CHECK: b.hi lbl     // encoding: [0bAAA01000,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.plast lbl
// CHECK: b.ls lbl     // encoding: [0bAAA01001,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.tcont lbl
// CHECK: b.ge lbl     // encoding: [0bAAA01010,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19

        b.tstop lbl
// CHECK: b.lt lbl     // encoding: [0bAAA01011,A,A,0x54]
// CHECK-NEXT:         //   fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch19
