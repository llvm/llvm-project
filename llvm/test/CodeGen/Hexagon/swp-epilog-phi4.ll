; RUN: llc -mtriple=hexagon -enable-pipeliner < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that we generate the correct value for a Phi in the epilog
; that is for a value defined two stages earlier. An extra copy in the
; epilog means the schedule is incorrect.

; CHECK: endloop0
; CHECK-NOT: r{{[0-9]+}} = r{{[0-9]+}}

; Function Attrs: nounwind
define void @f0(i32 %a0, ptr %a1, ptr %a2, ptr %a3, ptr %a4) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v8, %b1 ], [ 1, %b0 ]
  %v1 = load i32, ptr %a3, align 4, !tbaa !0
  %v2 = getelementptr inbounds i32, ptr %a1, i32 %v0
  %v3 = load i32, ptr %v2, align 4, !tbaa !0
  %v4 = load i32, ptr %a4, align 4, !tbaa !0
  %v5 = mul nsw i32 %v4, %v3
  %v6 = add nsw i32 %v5, %v1
  %v7 = getelementptr inbounds [1000 x i32], ptr %a2, i32 %v0, i32 0
  store i32 %v6, ptr %v7, align 4, !tbaa !0
  %v8 = add nsw i32 %v0, 1
  %v9 = icmp eq i32 %v8, %a0
  br i1 %v9, label %b2, label %b1

b2:                                               ; preds = %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }

!0 = !{!1, !1, i64 0}
!1 = !{!"long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
