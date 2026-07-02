; RUN: opt < %s -passes='loop-rotate<update-branch-weights>' -S | FileCheck %s
;
; Covers interaction between PreHeaderEntries and the ZeroTripCountWeights
; scale-up path. When header weights are small, updateBranchWeights scales
; OrigLoopExitWeight and OrigLoopBackedgeWeight (here ×128) to represent the
; zero-trip ratio. PreHeaderEntries must be scaled by the same factor; otherwise
; the rotated guard mixes scaled header weights with unscaled entry counts.
;
; Original:
;   entry  -> ph / ret   !prof {1, 1}     ; PreHeaderEntries = 1
;   header -> body/exit  !prof {100, 1}
;
; After scaling (×128):
;   PreHeaderEntries       = 128
;   OrigLoopExitWeight     = 128
;   OrigLoopBackedgeWeight = 12800
;
; Expected post-rotation:
;   GUARD !prof {1, 127}
;   LATCH !prof {127, 12673}
;
; Per-loop flow is conserved after scaling.

define void @small_scaled(ptr %p, i32 %n, i1 %c) !prof !0 {
entry:
  br i1 %c, label %ph, label %ret, !prof !1

ph:
  br label %header

header:
  %iv = phi i32 [ 0, %ph ], [ %next, %body ]
  %cmp = icmp slt i32 %iv, %n
  br i1 %cmp, label %body, label %exit, !prof !2

body:
  store volatile i32 %iv, ptr %p, align 4
  %next = add i32 %iv, 1
  br label %header

exit:
  ret void

ret:
  ret void
}

!0 = !{!"function_entry_count", i64 2}
!1 = !{!"branch_weights", i32 1, i32 1}
!2 = !{!"branch_weights", i32 100, i32 1}

; CHECK-LABEL: define void @small_scaled(

; The guard keeps the hot loop-entry edge hot after scaling PreHeaderEntries.
; CHECK:      ph:
; CHECK:        br i1 %{{.*}}, label %body.lr.ph, label %exit, !prof [[GUARD:![0-9]+]]

; The latch uses the same scaled unit as the guard and original header weights.
; CHECK:      body:
; CHECK:        br i1 %{{.*}}, label %body, label %header.exit_crit_edge, !prof [[LATCH:![0-9]+]]

; CHECK-DAG: [[GUARD]] = !{!"branch_weights", i32 127, i32 1}
; CHECK-DAG: [[LATCH]] = !{!"branch_weights", i32 12673, i32 127}
