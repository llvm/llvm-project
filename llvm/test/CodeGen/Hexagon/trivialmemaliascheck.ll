; RUN: llc -march=hexagon -enable-aa-sched-mi < %s | FileCheck %s

; The two memory addresses in the load and the memop below are trivially
; non-aliasing. However, there are some cases where the scheduler cannot
; determine this - in this case, it is because of the use of memops, that on the
; surface. do not have only one mem operand. However, the backend knows MIs and
; can step in and help some cases. In our case, if the base registers are the
; same and the offsets different and the memory access size is such that
; the two accesses won't overlap, we can tell the scheduler that there is no
; dependence due to aliasing between the two instructions.
; In the example below, this allows the load to be packetized with the memop.
; CHECK: {
; CHECK:      r{{[0-9]*}} = memw(r{{[0-9]*}}+#4)
; CHECK-NEXT: memw(r{{[0-9]*}}+#0) += #3
; CHECK: }

@g0 = common global [10 x i32] zeroinitializer, align 8

; Function Attrs: nounwind
define void @f0(i32* nocapture %a0) #0 {
b0:
  %v0 = load i32, i32* %a0, align 4, !tbaa !0
  %v1 = add nsw i32 %v0, 3
  store i32 %v1, i32* %a0, align 4, !tbaa !0
  %v2 = getelementptr inbounds i32, i32* %a0, i32 1
  %v3 = load i32, i32* %v2, align 4, !tbaa !0
  store i32 %v3, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @g0, i32 0, i32 0), align 8, !tbaa !0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
