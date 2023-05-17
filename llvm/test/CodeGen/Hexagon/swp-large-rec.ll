; RUN: llc -march=hexagon -enable-pipeliner -stats \
; RUN:     -pipeliner-prune-loop-carried=false -fp-contract=fast \
; RUN:     -o /dev/null < %s 2>&1 -pipeliner-experimental-cg=true | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts

; That that we do not pipeline this loop. The recurrence is too large. If
; we pipeline this example, that means we're not checking the complete
; chain of dependences.

; STATS-NOT: 1 pipeliner   - Number of loops software pipelined

; Function Attrs: nounwind
define void @f0(i32 %a0, i32 %a1, double %a2, double %a3, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11) #0 {
b0:
  br i1 undef, label %b1, label %b4

b1:                                               ; preds = %b3, %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi ptr [ %v22, %b2 ], [ %a5, %b1 ]
  %v1 = phi ptr [ %v23, %b2 ], [ %a6, %b1 ]
  %v2 = phi ptr [ %v24, %b2 ], [ %a7, %b1 ]
  %v3 = phi i32 [ %v20, %b2 ], [ 0, %b1 ]
  %v4 = load i8, ptr %v0, align 1, !tbaa !0
  %v5 = zext i8 %v4 to i32
  %v6 = load i8, ptr %v1, align 1, !tbaa !0
  %v7 = sext i8 %v6 to i32
  %v8 = load i8, ptr %v2, align 1, !tbaa !0
  %v9 = sext i8 %v8 to i32
  %v10 = mul nsw i32 %v9, %v7
  %v11 = add nsw i32 %v10, %v5
  %v12 = trunc i32 %v11 to i8
  store i8 %v12, ptr undef, align 1, !tbaa !0
  %v13 = load i8, ptr %v2, align 1, !tbaa !0
  %v14 = sext i8 %v13 to i32
  %v15 = load i8, ptr undef, align 1, !tbaa !0
  %v16 = sext i8 %v15 to i32
  %v17 = mul nsw i32 %v16, %v14
  %v18 = add i32 %v17, %v11
  %v19 = trunc i32 %v18 to i8
  store i8 %v19, ptr %v0, align 1, !tbaa !0
  %v20 = add nsw i32 %v3, 1
  store i8 0, ptr undef, align 1, !tbaa !0
  %v21 = icmp eq i32 %v20, undef
  %v22 = getelementptr i8, ptr %v0, i32 1
  %v23 = getelementptr i8, ptr %v1, i32 1
  %v24 = getelementptr i8, ptr %v2, i32 1
  br i1 %v21, label %b3, label %b2

b3:                                               ; preds = %b2
  tail call void @f1(i32 %a1, ptr %a4, ptr %a5, ptr %a6, ptr %a7, ptr %a8, ptr %a9, ptr %a10, ptr %a11, i8 signext 1) #2
  br i1 undef, label %b4, label %b1

b4:                                               ; preds = %b3, %b0
  ret void
}

declare void @f1(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8 signext) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { "target-cpu"="hexagonv55" }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
