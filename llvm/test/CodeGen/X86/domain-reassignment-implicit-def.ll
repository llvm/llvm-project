; RUN: llc -mcpu=skylake-avx512 -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s

; Check that the X86 Domain Reassignment pass doesn't drop IMPLICIT_DEF nodes,
; which would later cause crashes (e.g. in LiveVariables) - see PR37430
define void @domain_reassignment_implicit_def(i1 %cond, i8 *%mem, float %arg) {
; CHECK:    vxorps %xmm1, %xmm1, %xmm1
; CHECK:    vcmpneqss %xmm1, %xmm0, %k0
; CHECK:    kmovb %k0, (%rsi)
top:
  br i1 %cond, label %L19, label %L15

L15:                                              ; preds = %top
  %tmp47 = fcmp une float 0.000000e+00, %arg
  %tmp48 = zext i1 %tmp47 to i8
  br label %L21

L19:                                              ; preds = %top
  br label %L21

L21:                                              ; preds = %L19, %L15
  %.sroa.0.0 = phi i8 [ undef, %L19 ], [ %tmp48, %L15 ]
  store i8 %.sroa.0.0, i8* %mem, align 1
  ret void
}
