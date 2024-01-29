; RUN: opt < %s -tiny-trip-count-interleave-threshold=16 -force-target-max-vector-interleave=8 -p loop-vectorize -S -pass-remarks=loop-vectorize -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -tiny-trip-count-interleave-threshold=16 -force-target-max-vector-interleave=8 -p loop-vectorize -S 2>&1 | FileCheck %s -check-prefix=CHECK-IR
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

; This has the same trip count as loop_with_tc_32 but since the resulting interleaved group 
; in this case may access memory out-of-bounds, it requires a scalar epilogue iteration for 
; correctness, making at most 31 iterations available for interleaving.
; TODO: When the auto-vectorizer chooses VF 16, it should choose IC 1 to leave a smaller scalar remainder
; than IC 2
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_32_scalar_epilogue_reqd(ptr noalias %p, ptr noalias %q) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %gep.src = getelementptr inbounds [3 x i8], ptr %p, i64 %i, i64 0
  %l = load i8, ptr %gep.src, align 1
  %gep.dst = getelementptr inbounds i8, ptr %q, i64 %i
  store i8 %l, ptr %gep.dst, align 1
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

; For this loop with known TC of 48, when the auto-vectorizer chooses VF 16, it should choose
; IC 1 since there will be no remainder loop that needs to run after the vector loop.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 1)
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

; For this loop with known TC of 49, when the auto-vectorizer chooses VF 16, it should choose
; IC 1 since a remainder loop TC of 1 is more efficient than remainder loop TC of 17 with IC 2
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 1)
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

; For this loop with known TC of 55, when the auto-vectorizer chooses VF 16, it should choose
; IC 1 since a remainder loop TC of 7 is more efficient than remainder loop TC of 23 with IC 2
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 1)
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

; For this loop with known TC of 100, when the auto-vectorizer chooses VF 16, it should choose
; IC 2 since a remainder loop TC of 4 is more efficient than remainder loop TC of 36 with IC 4
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 2)
define void @loop_with_tc_100(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 100
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For this loop with known TC of 128, when the auto-vectorizer chooses VF 16, it should choose
; IC 8 since there is no remainder loop run needed after the vector loop runs
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 8)
define void @loop_with_tc_128(ptr noalias %p, ptr noalias %q) {
; CHECK-IR-LABEL: define void @loop_with_tc_128(
; CHECK-IR-SAME: ptr noalias [[P:%.*]], ptr noalias [[Q:%.*]]) {
; CHECK-IR-NEXT:  entry:
; CHECK-IR-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
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
  %cond = icmp eq i64 %i.next, 128
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; This has the same trip count as loop_with_tc_128 but since the resulting interleaved group 
; in this case may access memory out-of-bounds, it requires a scalar epilogue iteration for 
; correctness, making at most 127 iterations available for interleaving.
; TODO: Like loop_with_tc_128, the entry block should branch into the vector loop, instead of the scalar epilogue.
; TODO: When the auto-vectorizer chooses VF 16, it should choose IC 2 to leave a smaller scalar 
; remainder than IC 4
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 8)
define void @loop_with_tc_128_scalar_epilogue_reqd(ptr noalias %p, ptr noalias %q) {
; CHECK-IR-LABEL: define void @loop_with_tc_128_scalar_epilogue_reqd(
; CHECK-IR-SAME: ptr noalias [[P:%.*]], ptr noalias [[Q:%.*]]) {
; CHECK-IR-NEXT:  entry:
; CHECK-IR-NEXT:    br i1 true, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %gep.src = getelementptr inbounds [3 x i8], ptr %p, i64 %i, i64 0
  %l = load i8, ptr %gep.src, align 1
  %gep.dst = getelementptr inbounds i8, ptr %q, i64 %i
  store i8 %l, ptr %gep.dst, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 128
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For this loop with known TC of 129, when the auto-vectorizer chooses VF 16, it should choose
; IC 8 since there is a small remainder loop that needs to run after the vector loop
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 8)
define void @loop_with_tc_129(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 129
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For this loop with known TC of 180, when the auto-vectorizer chooses VF 16, it should choose
; IC 8 since the remainder loop of TC 52 cannot be reduced by choosing IC 4
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 8)
define void @loop_with_tc_180(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 180
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For this loop with known TC of 193, when the auto-vectorizer chooses VF 16, it should choose
; IC 4 since a remainder loop TC of 1 is more efficient than remainder loop TC of 65 with IC 8
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 4)
define void @loop_with_tc_193(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 193
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; For a loop with high known TC of 1000, when the auto-vectorizer chooses VF 16, the IC will
; be capped by the target-specific maximum interleave count
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 16, interleaved count: 8)
define void @loop_with_tc_1000(ptr noalias %p, ptr noalias %q) {
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
  %cond = icmp eq i64 %i.next, 1000
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}
