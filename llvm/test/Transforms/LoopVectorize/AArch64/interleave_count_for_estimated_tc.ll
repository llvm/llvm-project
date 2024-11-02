; RUN: opt < %s -tiny-trip-count-interleave-threshold=16 -force-target-max-vector-interleave=8 -p loop-vectorize -S -pass-remarks=loop-vectorize -disable-output 2>&1 | FileCheck %s
; TODO: remove -tiny-trip-count-interleave-threshold once the interleave threshold is removed

target triple = "aarch64-linux-gnu"

%pair = type { i8, i8 }

; For a loop with a profile-guided estimated TC of 32, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 1)
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

; For a loop with a profile-guided estimated TC of 33, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 1)
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

; For a loop with a profile-guided estimated TC of 48, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 1)
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

; For a loop with a profile-guided estimated TC of 63, when the auto-vectorizer chooses VF 16, 
; it should conservatively choose IC 1 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 1)
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

; For a loop with a profile-guided estimated TC of 100, when the auto-vectorizer chooses VF 16, 
; it should choose conservatively IC 2 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_profile_tc_100(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !5

for.end:
  ret void
}

; For a loop with a profile-guided estimated TC of 128, when the auto-vectorizer chooses VF 16, 
; it should choose conservatively IC 4 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 4)
define void @loop_with_profile_tc_128(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !6

for.end:
  ret void
}

; For a loop with a profile-guided estimated TC of 129, when the auto-vectorizer chooses VF 16, 
; it should choose conservatively IC 4 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 4)
define void @loop_with_profile_tc_129(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !7

for.end:
  ret void
}

; For a loop with a profile-guided estimated TC of 180, when the auto-vectorizer chooses VF 16, 
; it should choose conservatively IC 4 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 4)
define void @loop_with_profile_tc_180(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !8

for.end:
  ret void
}

; For a loop with a profile-guided estimated TC of 193, when the auto-vectorizer chooses VF 16, 
; it should choose conservatively IC 4 so that the vector loop runs twice at least
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 4)
define void @loop_with_profile_tc_193(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !9

for.end:
  ret void
}

; For a loop with a profile-guided estimated TC of 1000, when the auto-vectorizer chooses VF 16, 
; the IC will be capped by the target-specific maximum interleave count
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 8)
define void @loop_with_profile_tc_1000(ptr noalias %p, ptr noalias %q, i64 %n) {
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
  br i1 %cond, label %for.end, label %for.body, !prof !10

for.end:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 31}
!1 = !{!"branch_weights", i32 1, i32 32}
!2 = !{!"branch_weights", i32 1, i32 47}
!3 = !{!"branch_weights", i32 1, i32 62}
!4 = !{!"branch_weights", i32 1, i32 63}
!5 = !{!"branch_weights", i32 1, i32 99}
!6 = !{!"branch_weights", i32 1, i32 127}
!7 = !{!"branch_weights", i32 1, i32 128}
!8 = !{!"branch_weights", i32 1, i32 179}
!9 = !{!"branch_weights", i32 1, i32 192}
!10 = !{!"branch_weights", i32 1, i32 999}
