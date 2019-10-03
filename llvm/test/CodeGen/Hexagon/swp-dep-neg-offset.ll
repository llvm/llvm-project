; RUN: llc -march=hexagon -enable-pipeliner -hexagon-initial-cfg-cleanup=0 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that the code that changes the dependences does not allow
; a load with a negative offset to be overlapped with the post
; increment store that generates the base register.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: = mem{{u?}}b([[REG:(r[0-9])+]]+#-1)
; CHECK-NOT: memb([[REG]]{{\+?}}#0) =
; CHECK: }
; CHECK: }{{[ \t]*}}:endloop0

@g0 = external global [1000000 x i8], align 8

; Function Attrs: nounwind
define void @f0(i32 %a0, [1000 x i8]* %a1, [1000 x i8]* %a2) #0 {
b0:
  br i1 undef, label %b1, label %b7

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b6

b2:                                               ; preds = %b5, %b1
  br i1 undef, label %b3, label %b5

b3:                                               ; preds = %b3, %b2
  %v0 = phi i32 [ %v17, %b3 ], [ 1, %b2 ]
  %v1 = phi i32 [ %v16, %b3 ], [ 0, %b2 ]
  %v2 = add nsw i32 %v0, -1
  %v3 = getelementptr inbounds [1000 x i8], [1000 x i8]* %a1, i32 undef, i32 %v2
  %v4 = load i8, i8* %v3, align 1, !tbaa !0
  %v5 = zext i8 %v4 to i32
  %v6 = getelementptr inbounds [1000000 x i8], [1000000 x i8]* @g0, i32 0, i32 %v1
  %v7 = load i8, i8* %v6, align 1, !tbaa !0
  %v8 = sext i8 %v7 to i32
  %v9 = getelementptr inbounds [1000 x i8], [1000 x i8]* %a2, i32 undef, i32 %v0
  %v10 = load i8, i8* %v9, align 1, !tbaa !0
  %v11 = sext i8 %v10 to i32
  %v12 = mul nsw i32 %v11, %v8
  %v13 = add nsw i32 %v12, %v5
  %v14 = trunc i32 %v13 to i8
  %v15 = getelementptr inbounds [1000 x i8], [1000 x i8]* %a1, i32 undef, i32 %v0
  store i8 %v14, i8* %v15, align 1, !tbaa !0
  %v16 = add nsw i32 %v1, 1
  %v17 = add nsw i32 %v0, 1
  %v18 = icmp eq i32 %v17, %a0
  br i1 %v18, label %b4, label %b3

b4:                                               ; preds = %b3
  br label %b5

b5:                                               ; preds = %b4, %b2
  br i1 undef, label %b6, label %b2

b6:                                               ; preds = %b5, %b1
  unreachable

b7:                                               ; preds = %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
