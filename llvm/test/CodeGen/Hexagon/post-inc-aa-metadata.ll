; RUN: llc -march=hexagon -debug-only=isel < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Check that the generated post-increment load has TBAA information.
; CHECK-LABEL: Machine code for function fred:
; CHECK: = V6_vL32b_pi %vreg{{[0-9]+}}<tied1>, 64; mem:LD64[{{.*}}](tbaa=

target triple = "hexagon"

; Function Attrs: norecurse nounwind
define void @fred(<16 x i32>* nocapture %p, <16 x i32>* nocapture readonly %q, i32 %n) local_unnamed_addr #0 {
entry:
  %tobool2 = icmp eq i32 %n, 0
  br i1 %tobool2, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %n.addr.05 = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %q.addr.04 = phi <16 x i32>* [ %incdec.ptr, %while.body ], [ %q, %entry ]
  %p.addr.03 = phi <16 x i32>* [ %incdec.ptr1, %while.body ], [ %p, %entry ]
  %dec = add i32 %n.addr.05, -1
  %incdec.ptr = getelementptr inbounds <16 x i32>, <16 x i32>* %q.addr.04, i32 1
  %0 = load <16 x i32>, <16 x i32>* %q.addr.04, align 64, !tbaa !1
  %incdec.ptr1 = getelementptr inbounds <16 x i32>, <16 x i32>* %p.addr.03, i32 1
  store <16 x i32> %0, <16 x i32>* %p.addr.03, align 64, !tbaa !1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,-hvx-double" }


!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
