; RUN: opt < %s -tiny-trip-count-interleave-threshold=16 -p loop-vectorize -S -pass-remarks=loop-vectorize -disable-output 2>&1 | FileCheck %s
; TODO: remove -tiny-trip-count-interleave-threshold once the interleave threshold is removed

target triple = "aarch64-linux-gnu"

%pair = type { i8, i8 }

; For this loop with known TC of 32, when the auto-vectorizer chooses VF 16, it should choose
; IC 2 since there is no remainder loop run needed after the vector loop runs.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_32(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 32
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For this loop with known TC of 33, when the auto-vectorizer chooses VF 16, it should choose
; IC 2 since there is a small remainder loop TC that needs to run after the vector loop.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_33(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 33
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For this loop with known TC of 39, when the auto-vectorizer chooses VF 16, it should choose
; IC 2 since there is a small remainder loop that needs to run after the vector loop.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_39(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 39
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; TODO: For this loop with known TC of 48, when the auto-vectorizer chooses VF 16, it should choose
; IC 1 since there will be no remainder loop that needs to run after the vector loop.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_48(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 48
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; TODO: For this loop with known TC of 49, when the auto-vectorizer chooses VF 16, it should choose
; IC 1 since a remainder loop TC of 1 is more efficient than remainder loop TC of 17 with IC 2
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_49(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 49
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; TODO: For this loop with known TC of 55, when the auto-vectorizer chooses VF 16, it should choose
; IC 1 since a remainder loop TC of 7 is more efficient than remainder loop TC of 23 with IC 2
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_55(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 55
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; TODO: For a loop with a profile-guided estimated TC of 32, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_profile_tc_32(ptr noalias %p, ptr noalias %q, i64 %n) {
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

; TODO: For a loop with a profile-guided estimated TC of 33, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_profile_tc_33(ptr noalias %p, ptr noalias %q, i64 %n) {
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

; TODO: For a loop with a profile-guided estimated TC of 48, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_profile_tc_48(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !2

for.end:
  ret void
}

; TODO: For a loop with a profile-guided estimated TC of 63, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_profile_tc_63(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !3

for.end:
  ret void
}

; For a loop with a profile-guided estimated TC of 64, when the auto-vectorizer chooses VF 16, 
; it should choose conservatively IC 2 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_profile_tc_64(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !4

for.end:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 31}
!1 = !{!"branch_weights", i32 1, i32 32}
!2 = !{!"branch_weights", i32 1, i32 47}
!3 = !{!"branch_weights", i32 1, i32 62}
!4 = !{!"branch_weights", i32 1, i32 63}
