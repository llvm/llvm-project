; This test shows the evolution of RVV pseudo instructions within isel.

; RUN: llc -mtriple riscv64 -mattr=+experimental-v %s -o %t.pre.mir \
; RUN:     -stop-before=finalize-isel
; RUN: cat %t.pre.mir | FileCheck --check-prefix=PRE-INSERTER %s

; RUN: llc -mtriple riscv64 -mattr=+experimental-v %t.pre.mir -o %t.post.mir \
; RUN:     -start-before=finalize-isel -stop-after=finalize-isel
; RUN: cat %t.post.mir | FileCheck --check-prefix=POST-INSERTER %s

define void @vadd_vint64m1(
          <vscale x 1 x i64> *%pc,
          <vscale x 1 x i64> *%pa,
          <vscale x 1 x i64> *%pb)
{
  %va = load <vscale x 1 x i64>, <vscale x 1 x i64>* %pa
  %vb = load <vscale x 1 x i64>, <vscale x 1 x i64>* %pb
  %vc = add <vscale x 1 x i64> %va, %vb
  store <vscale x 1 x i64> %vc, <vscale x 1 x i64> *%pc
  ret void
}

; PRE-INSERTER: %3:vr = VL1RE64_V %1 :: (load unknown-size from %ir.pa, align 8)
; PRE-INSERTER: %4:vr = VL1RE64_V %2 :: (load unknown-size from %ir.pb, align 8)
; PRE-INSERTER: %5:vr = PseudoVADD_VV_M1 killed %3, killed %4, $x0, 6, implicit $vl, implicit $vtype
; PRE-INSERTER:  VS1R_V killed %5, %0 :: (store unknown-size into %ir.pc, align 8)

; POST-INSERTER: %3:vr = VL1RE64_V %1 :: (load unknown-size from %ir.pa, align 8)
; POST-INSERTER: %4:vr = VL1RE64_V %2 :: (load unknown-size from %ir.pb, align 8)
; POST-INSERTER: dead %6:gpr = PseudoVSETVLI $x0, 88, implicit-def $vl, implicit-def $vtype
; POST-INSERTER: %5:vr = PseudoVADD_VV_M1 killed %3, killed %4, $noreg, 6, implicit $vl, implicit $vtype
; POST-INSERTER: VS1R_V killed %5, %0 :: (store unknown-size into %ir.pc, align 8)
