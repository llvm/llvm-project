; RUN: llc -march=hexagon -O2 -relocation-model=pic < %s | FileCheck %s
; CHECK: add({{pc|PC}},##_GLOBAL_OFFSET_TABLE_@PCREL)
; CHECK: call g1@GDPLT
; CHECK: call g0@GDPLT

@g0 = external thread_local global i32
@g1 = external thread_local global i32

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = load i32, ptr @g1, align 4, !tbaa !0
  store i32 %v0, ptr @g0, align 4, !tbaa !0
  tail call void @f1(i32 %v0) #0
  ret i32 0
}

declare void @f1(i32)

attributes #0 = { nounwind "target-cpu"="hexagonv5" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
