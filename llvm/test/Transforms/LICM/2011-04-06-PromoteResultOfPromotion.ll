; RUN: opt < %s -passes=licm -S | FileCheck %s
; PR9634

@g_58 = common global i32 0, align 4
@g_116 = common global ptr null, align 8

define void @f() nounwind {

; CHECK: entry:
; CHECK: alloca [9 x i16]
; CHECK: load i32, ptr @g_58
; CHECK: br label %for.body

entry:
  %l_87.i = alloca [9 x i16], align 16
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %inc12 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  store ptr @g_58, ptr @g_116, align 8, !tbaa !0
  %tmp2 = load ptr, ptr @g_116, align 8, !tbaa !0
  %tmp3 = load i32, ptr %tmp2, !tbaa !4
  %or = or i32 %tmp3, 10
  store i32 %or, ptr %tmp2, !tbaa !4
  %inc = add nsw i32 %inc12, 1
  %cmp = icmp slt i32 %inc, 4
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

!0 = !{!5, !5, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"short", !1}
!4 = !{!6, !6, i64 0}
!5 = !{!"any pointer", !1}
!6 = !{!"int", !1}
