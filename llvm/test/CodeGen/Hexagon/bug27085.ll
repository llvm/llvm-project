; RUN: llc -march=hexagon -relocation-model=pic -mattr=+long-calls < %s | FileCheck --check-prefix=CHECK-LONG %s
; RUN: llc -march=hexagon -relocation-model=pic < %s | FileCheck %s

; CHECK-LONG: call ##g0@GDPLT
; CHECK-LONG-NOT: call g0@GDPLT
; CHECK: call g0@GDPLT
; CHECK-NOT: call ##g0@GDPLT

target triple = "hexagon--linux"

@g0 = internal thread_local global i32 0, align 4

; Function Attrs: norecurse nounwind
define void @f0(i32 %a0) local_unnamed_addr #0 {
b0:
  store volatile i32 1, i32* @g0, align 4, !tbaa !1
  ret void
}

; Function Attrs: norecurse nounwind
define zeroext i1 @f1() local_unnamed_addr #0 {
b0:
  %v0 = load volatile i32, i32* @g0, align 4, !tbaa !1
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %b2, label %b1

b1:                                               ; preds = %b0
  store volatile i32 0, i32* @g0, align 4, !tbaa !1
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v2 = phi i1 [ true, %b1 ], [ false, %b0 ]
  ret i1 %v2
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"PIC Level", i32 1}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
