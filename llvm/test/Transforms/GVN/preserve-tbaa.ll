; RUN: opt -tbaa -basicaa -gvn -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

; GVN should preserve the TBAA tag on loads when doing PRE.

; CHECK: @test
; CHECK: %tmp33.pre = load i16* %P, align 2, !tbaa !0
; CHECK: br label %for.body
define void @test(i16 *%P, i16* %Q) nounwind {
entry:
  br i1 undef, label %bb.nph, label %for.end

bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %bb.nph
  %tmp33 = load i16* %P, align 2, !tbaa !0
  store i16 %tmp33, i16* %Q

  store i16 0, i16* %P, align 2, !tbaa !0
  br i1 false, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

!0 = metadata !{metadata !"short", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
