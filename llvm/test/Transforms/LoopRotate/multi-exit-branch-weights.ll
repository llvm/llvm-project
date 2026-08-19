; RUN: opt < %s -passes=loop-rotate -S | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefix=BFI-BEFORE
; RUN: opt < %s -passes='loop(loop-rotate),print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefix=BFI-AFTER
;
; The copied guard and latch run the same test. Keep the original weights on
; both, including loops with another body exit.
;
; Entry weights use another scale because each branch stores a local ratio.
;
;   entry --(c0)--> ph        !prof { 42, 2 }
;   ph    --------> header
;   header --(hcmp)--> exit1 / body   !prof { 200, 9800 }
;   body   --(bcmp)--> exit2 / latch  !prof { 10, 9790 }
;   latch  ---------> header

define void @f(ptr %p, i32 %start, i32 %limit, i32 %cond) !prof !3 {
entry:
  %c0 = icmp ne i32 %cond, 0
  br i1 %c0, label %ph, label %ret, !prof !0

ph:                                               ; preds = %entry
  br label %header

header:                                           ; preds = %latch, %ph
  %iv = phi i32 [ %start, %ph ], [ %iv.next, %latch ]
  ; Keep the guard conditional.
  %hcmp = icmp eq i32 %iv, %limit
  br i1 %hcmp, label %exit1, label %body, !prof !1

body:                                             ; preds = %header
  %addr = getelementptr i32, ptr %p, i32 %iv
  %v = load i32, ptr %addr, align 4
  %bcmp = icmp slt i32 %v, 0
  br i1 %bcmp, label %exit2, label %latch, !prof !2

latch:                                            ; preds = %body
  %iv.next = add i32 %iv, 1
  br label %header

exit1:                                            ; preds = %header
  ret void

exit2:                                            ; preds = %body
  ret void

ret:                                              ; preds = %entry
  ret void
}

; Keep explicit trip-count metadata for multi-exit loops. Rotation decrements
; the attribute but does not infer a trip count from branch weights without it.
define void @explicit_trip_count(ptr %p, i32 %start, i32 %limit) {
entry:
  br label %header

header:
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %latch ]
  %hcmp = icmp eq i32 %iv, %limit
  br i1 %hcmp, label %exit1, label %body, !prof !1, !llvm.loop !4

body:
  %v = load i32, ptr %p, align 4
  %bcmp = icmp slt i32 %v, 0
  br i1 %bcmp, label %exit2, label %latch, !prof !2

latch:
  %iv.next = add i32 %iv, 1
  br label %header

exit1:
  ret void

exit2:
  ret void
}

!0 = !{!"branch_weights", i32 42, i32 2}
!1 = !{!"branch_weights", i32 200, i32 9800}
!2 = !{!"branch_weights", i32 10, i32 9790}
!3 = !{!"function_entry_count", i64 44}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.estimated_trip_count", i32 10}

; BFI-BEFORE-LABEL: block-frequency-info: f
; BFI-BEFORE: - body: {{.*}} count = 1960
; BFI-BEFORE: - exit1: {{.*}} count = 40
; BFI-BEFORE: - exit2: {{.*}} count = 2
; BFI-BEFORE: - ret: {{.*}} count = 2

; BFI-AFTER-LABEL: block-frequency-info: f
; BFI-AFTER: - body: {{.*}} count = 1960
; BFI-AFTER: - exit1: {{.*}} count = 40
; BFI-AFTER: - exit2: {{.*}} count = 2
; BFI-AFTER: - ret: {{.*}} count = 2

; CHECK-LABEL: define void @f(

; CHECK:      ph:
; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %{{.*}}.lr.ph, !prof [[WEIGHTS:![0-9]+]]

; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %body, !prof [[WEIGHTS]]{{$}}

; CHECK-LABEL: define void @explicit_trip_count(
; CHECK: br i1 %{{.*}}, label %{{.*}}, label %body, !prof {{![0-9]+}}, !llvm.loop [[EXPLICIT_LOOP:![0-9]+]]
; CHECK-DAG: [[WEIGHTS]] = !{!"branch_weights", i32 200, i32 9800}
; CHECK-DAG: [[EXPLICIT_LOOP]] = distinct !{[[EXPLICIT_LOOP]], [[EXPLICIT_TC:![0-9]+]]}
; CHECK-DAG: [[EXPLICIT_TC]] = !{!"llvm.loop.estimated_trip_count", i32 9}
