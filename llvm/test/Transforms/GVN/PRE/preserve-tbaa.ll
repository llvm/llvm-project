; RUN: opt -passes=gvn -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

; GVN should preserve the TBAA tag on loads when doing PRE.

; CHECK-LABEL: @test(
; CHECK: %tmp33.pre = load i16, ptr %P, align 2, !tbaa !0
; CHECK: br label %for.body
define void @test(ptr %P, ptr %Q) nounwind {
entry:
  br i1 undef, label %bb.nph, label %for.end

bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %bb.nph
  %tmp33 = load i16, ptr %P, align 2, !tbaa !0
  store i16 %tmp33, ptr %Q

  store i16 0, ptr %P, align 2, !tbaa !0
  br i1 false, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

!0 = !{!3, !3, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"short", !1}
