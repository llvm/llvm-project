; RUN: llc < %s -o /dev/null "-mtriple=thumbv7-apple-ios" -debug-only=post-RA-sched 2> %t
; RUN: FileCheck %s < %t
; REQUIRES: asserts
; Make sure that mayalias store-load dependencies have one cycle
; latency regardless of whether they are barriers or not.

; CHECK: ** List Scheduling
; CHECK: SU(2){{.*}}STR{{.*}}Volatile
; CHECK-NOT: ch SU
; CHECK: ch SU(3): Latency=1
; CHECK-NOT: ch SU
; CHECK: SU(3){{.*}}LDR{{.*}}Volatile
; CHECK-NOT: ch SU
; CHECK: ch SU(2): Latency=1
; CHECK-NOT: ch SU
; CHECK: ** List Scheduling
; CHECK: SU(2){{.*}}STR{{.*}}
; CHECK-NOT: ch SU
; CHECK: ch SU(3): Latency=1
; CHECK-NOT: ch SU
; CHECK: SU(3){{.*}}LDR{{.*}}
; CHECK-NOT: ch SU
; CHECK: ch SU(2): Latency=1
; CHECK-NOT: ch SU
define i32 @f1(i32* nocapture %p1, i32* nocapture %p2) nounwind {
entry:
  store volatile i32 65540, i32* %p1, align 4, !tbaa !0
  %0 = load volatile i32* %p2, align 4, !tbaa !0
  ret i32 %0
}

define i32 @f2(i32* nocapture %p1, i32* nocapture %p2) nounwind {
entry:
  store i32 65540, i32* %p1, align 4, !tbaa !0
  %0 = load i32* %p2, align 4, !tbaa !0
  ret i32 %0
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
