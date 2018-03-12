; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts
; Expect successful compilation.

target triple = "hexagon"

; Function Attrs: nounwind optsize
define void @f0(i32* nocapture %a0, i8* %a1) #0 {
b0:
  call void @llvm.hexagon.prefetch(i8* %a1)
  store i32 0, i32* %a0, align 4, !tbaa !0
  ret void
}

; Function Attrs: nounwind
declare void @llvm.hexagon.prefetch(i8*) #1

attributes #0 = { nounwind optsize "target-cpu"="hexagonv55" }
attributes #1 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
