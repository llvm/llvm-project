; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; Checking for alignment of stack to 64.
; CHECK: r{{[0-9]+}} = and(r{{[0-9]+}},#-64)

target triple = "hexagon"

%s.0 = type { i32, i32, i32, i32, i32 }

@g0 = private unnamed_addr constant [7 x i8] c"%x %x\0A\00", align 8
@g1 = global %s.0 { i32 11, i32 13, i32 15, i32 17, i32 19 }, align 4
@g2 = global <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>, align 64

; Function Attrs: nounwind
declare i32 @f0(ptr nocapture, ...) #0

; Function Attrs: nounwind
define void @f1(ptr byval(%s.0) %a0, <16 x i32> %a1) #0 {
b0:
  %v0 = alloca <16 x i32>, align 64
  store <16 x i32> %a1, ptr %v0, align 64, !tbaa !0
  %v1 = ptrtoint ptr %a0 to i32
  %v2 = ptrtoint ptr %v0 to i32
  %v3 = call i32 (ptr, ...) @f0(ptr @g0, i32 %v1, i32 %v2) #0
  ret void
}

; Function Attrs: nounwind
define i32 @f2() #0 {
b0:
  %v0 = load <16 x i32>, ptr @g2, align 64, !tbaa !0
  tail call void @f1(ptr byval(%s.0) @g1, <16 x i32> %v0)
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
