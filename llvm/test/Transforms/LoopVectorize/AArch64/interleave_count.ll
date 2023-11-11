; RUN: opt < %s -force-vector-width=8 -tiny-trip-count-interleave-threshold=16 -p loop-vectorize -S -pass-remarks=loop-vectorize 2>&1 | FileCheck %s
; TODO: remove -tiny-trip-count-interleave-threshold once the interleave threshold is removed

target triple = "aarch64-linux-gnu"

%pair = type { i8, i8 }

; For a loop with known trip count of 16, when we force VF 8, it should use
; IC 2 since there is no remainder loop run needed when the vector loop runs.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 8, interleaved count: 2)
define void @loop_with_tc_16(ptr noalias %p, ptr noalias %q) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr %pair, ptr %p, i64 %i, i32 0
  %tmp1 = load i8, ptr %tmp0, align 1
  %tmp2 = getelementptr %pair, ptr %p, i64 %i, i32 1
  %tmp3 = load i8, ptr %tmp2, align 1
  %add = add i8 %tmp1, %tmp3
  %qi = getelementptr i8, ptr %q, i64 %i
  store i8 %add, ptr %qi, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 16
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; TODO: For a loop with known trip count of 17, when we force VF 8, it should use
; IC 1 since there may be a remainder loop that needs to run after the vector loop.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 8, interleaved count: 1)
define void @loop_with_tc_17(ptr noalias %p, ptr noalias %q) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr %pair, ptr %p, i64 %i, i32 0
  %tmp1 = load i8, ptr %tmp0, align 1
  %tmp2 = getelementptr %pair, ptr %p, i64 %i, i32 1
  %tmp3 = load i8, ptr %tmp2, align 1
  %add = add i8 %tmp1, %tmp3
  %qi = getelementptr i8, ptr %q, i64 %i
  store i8 %add, ptr %qi, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 17
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For a loop with unknown trip count but a profile showing an approx TC estimate of 16, 
; when we force VF 8, it should use IC 2 since chances are high that the remainder loop
; won't need to run
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 8, interleaved count: 2)
define void @loop_with_profile_tc_16(ptr noalias %p, ptr noalias %q, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr %pair, ptr %p, i64 %i, i32 0
  %tmp1 = load i8, ptr %tmp0, align 1
  %tmp2 = getelementptr %pair, ptr %p, i64 %i, i32 1
  %tmp3 = load i8, ptr %tmp2, align 1
  %add = add i8 %tmp1, %tmp3
  %qi = getelementptr i8, ptr %q, i64 %i
  store i8 %add, ptr %qi, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %for.end, label %for.body, !prof !0

for.end:
  ret void
}

; TODO: For a loop with unknown trip count but a profile showing an approx TC estimate of 17, 
; when we force VF 8, it should use IC 1 since chances are high that the remainder loop
; will need to run
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 8, interleaved count: 1)
define void @loop_with_profile_tc_17(ptr noalias %p, ptr noalias %q, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr %pair, ptr %p, i64 %i, i32 0
  %tmp1 = load i8, ptr %tmp0, align 1
  %tmp2 = getelementptr %pair, ptr %p, i64 %i, i32 1
  %tmp3 = load i8, ptr %tmp2, align 1
  %add = add i8 %tmp1, %tmp3
  %qi = getelementptr i8, ptr %q, i64 %i
  store i8 %add, ptr %qi, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %for.end, label %for.body, !prof !1

for.end:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 15}
!1 = !{!"branch_weights", i32 1, i32 16}
