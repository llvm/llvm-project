; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; CHECK: r{{[0-9]*}} = add(##g0,asl(r{{[0-9]*}},#2))

%s.0 = type { i16, i8 }

@g0 = internal global [20 x ptr] zeroinitializer, align 8

; Function Attrs: nounwind
define void @f0(ptr %a0) #0 {
b0:
  %v0 = icmp eq ptr %a0, null
  br i1 %v0, label %b2, label %b1

b1:                                               ; preds = %b0
  %v1 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 1
  %v2 = load i8, ptr %v1, align 1, !tbaa !0
  %v3 = zext i8 %v2 to i32
  %v4 = getelementptr inbounds [20 x ptr], ptr @g0, i32 0, i32 %v3
  tail call void @f1(ptr %v4) #0
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

declare void @f1(ptr)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
