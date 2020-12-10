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

; PRE-INSERTER: %4:vr = IMPLICIT_DEF
; PRE-INSERTER: %3:vr = PseudoVLE64_V_M1 %4, %1, $noreg, $x0, 64, implicit $vl, implicit $vtype :: (load unknown-size from %ir.pa, align 8)
; PRE-INSERTER: %6:vr = IMPLICIT_DEF
; PRE-INSERTER: %5:vr = PseudoVLE64_V_M1 %6, %2, $noreg, $x0, 64, implicit $vl, implicit $vtype :: (load unknown-size from %ir.pb, align 8)
; PRE-INSERTER: %8:vr = IMPLICIT_DEF
; PRE-INSERTER: %7:vr = PseudoVADD_VV_M1 %8, killed %3, killed %5, $noreg, $x0, 64, implicit $vl, implicit $vtype
; PRE-INSERTER: PseudoVSE64_V_M1 killed %7, %0, $noreg, $x0, 64, implicit $vl, implicit $vtype :: (store unknown-size into %ir.pc, align 8)

; POST-INSERTER: %4:vr = IMPLICIT_DEF
; POST-INSERTER: dead %9:gpr = PseudoVSETVLI $x0, 12, implicit-def $vl, implicit-def $vtype
; POST-INSERTER: %3:vr = PseudoVLE64_V_M1 %4, %1, $noreg, $noreg, -1, implicit $vl, implicit $vtype :: (load unknown-size from %ir.pa, align 8)
; POST-INSERTER: %6:vr = IMPLICIT_DEF
; POST-INSERTER: dead %10:gpr = PseudoVSETVLI $x0, 12, implicit-def $vl, implicit-def $vtype
; POST-INSERTER: %5:vr = PseudoVLE64_V_M1 %6, %2, $noreg, $noreg, -1, implicit $vl, implicit $vtype :: (load unknown-size from %ir.pb, align 8)
; POST-INSERTER: %8:vr = IMPLICIT_DEF
; POST-INSERTER: dead %11:gpr = PseudoVSETVLI $x0, 12, implicit-def $vl, implicit-def $vtype
; POST-INSERTER: %7:vr = PseudoVADD_VV_M1 %8, killed %3, killed %5, $noreg, $noreg, -1, implicit $vl, implicit $vtype
; POST-INSERTER: dead %12:gpr = PseudoVSETVLI $x0, 12, implicit-def $vl, implicit-def $vtype
; POST-INSERTER: PseudoVSE64_V_M1 killed %7, %0, $noreg, $noreg, -1, implicit $vl, implicit $vtype :: (store unknown-size into %ir.pc, align 8)
