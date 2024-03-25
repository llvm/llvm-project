; RUN: opt < %s -passes=loop-vectorize,dce -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -force-vector-width=4 -force-vector-interleave=0 -S \
; RUN:   | FileCheck %s --check-prefix=CHECK-VECTOR
; RUN: opt < %s -passes=loop-vectorize,dce -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -force-vector-width=1 -force-vector-interleave=0 -S \
; RUN:   | FileCheck %s --check-prefix=CHECK-SCALAR

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; We don't unroll this loop because it has a small constant trip count 
; that is not profitable for generating a scalar epilogue
;
; CHECK-VECTOR-LABEL: @foo_trip_count_8(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo_trip_count_8(
; CHECK-SCALAR: load i32, ptr
; CHECK-SCALAR-NOT: load i32, ptr
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @foo_trip_count_8(ptr nocapture %A) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %0, align 4
  %2 = add nsw i32 %1, 6
  store i32 %2, ptr %0, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 8
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                       ; preds = %for.body
  ret void
}

; We should unroll this loop 4 times since TC being a multiple of VF means
; that the epilogue loop may not need to run, making it profitable for
; the vector loop to run even once
;
; CHECK-VECTOR-LABEL: @foo_trip_count_16(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo_trip_count_16(
; CHECK-SCALAR: load i32, ptr
; CHECK-SCALAR-NOT: load i32, ptr
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @foo_trip_count_16(ptr nocapture %A) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %0, align 4
  %2 = add nsw i32 %1, 6
  store i32 %2, ptr %0, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                       ; preds = %for.body
  ret void
}

; We should unroll this loop four times since unrolling it twice
; will produce the same epilogue TC of 1, making larger unroll count
; more profitable
;
; CHECK-VECTOR-LABEL: @foo_trip_count_17(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo_trip_count_17(
; CHECK-SCALAR: load i32, ptr
; CHECK-SCALAR-NOT: load i32, ptr
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @foo_trip_count_17(ptr nocapture %A) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %0, align 4
  %2 = add nsw i32 %1, 6
  store i32 %2, ptr %0, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 17
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                       ; preds = %for.body
  ret void
}

; We should unroll this loop twice since unrolling four times will 
; create an epilogue loop of TC 8, while unrolling it twice will 
; eliminate the epologue loop altogether
;
; CHECK-VECTOR-LABEL: @foo_trip_count_24(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo_trip_count_24(
; CHECK-SCALAR: load i32, ptr
; CHECK-SCALAR-NOT: load i32, ptr
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @foo_trip_count_24(ptr nocapture %A) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %0, align 4
  %2 = add nsw i32 %1, 6
  store i32 %2, ptr %0, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 24
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                       ; preds = %for.body
  ret void
}

; We should unroll this loop twice since TC not being a multiple of VF may require
; the epilogue loop to run, making it profitable when the vector loop runs
; at least twice.
;
; CHECK-VECTOR-LABEL: @foo_trip_count_25(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo_trip_count_25(
; CHECK-SCALAR: load i32, ptr
; CHECK-SCALAR-NOT: load i32, ptr
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @foo_trip_count_25(ptr nocapture %A) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %0, align 4
  %2 = add nsw i32 %1, 6
  store i32 %2, ptr %0, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 25
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                       ; preds = %for.body
  ret void
}

; We should unroll this loop 4 times since TC not being a multiple of VF may require
; the epilogue loop to run, making it profitable when the vector loop runs
; at least twice.
;
; CHECK-VECTOR-LABEL: @foo_trip_count_33(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo_trip_count_33(
; CHECK-SCALAR: load i32, ptr
; CHECK-SCALAR-NOT: load i32, ptr
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @foo_trip_count_33(ptr nocapture %A) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %0, align 4
  %2 = add nsw i32 %1, 6
  store i32 %2, ptr %0, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 33
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                       ; preds = %for.body
  ret void
}

; We should unroll this loop 4 times since TC not being a multiple of VF may require
; the epilogue loop to run, making it profitable when the vector loop runs
; at least twice. The IC is restricted to 4 since that is the maximum supported 
; for the target.
;
; CHECK-VECTOR-LABEL: @foo_trip_count_101(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo_trip_count_101(
; CHECK-SCALAR: load i32, ptr
; CHECK-SCALAR-NOT: load i32, ptr
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @foo_trip_count_101(ptr nocapture %A) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                       ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %0, align 4
  %2 = add nsw i32 %1, 6
  store i32 %2, ptr %0, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 101
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                       ; preds = %for.body
  ret void
}

; But this is a good small loop to unroll as we don't know of a bound on its
; trip count.
;
; CHECK-VECTOR-LABEL: @bar(
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: ret
;
; For x86, loop unroll in loop vectorizer is disabled when VF==1.
;
; CHECK-SCALAR-LABEL: @bar(
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define void @bar(ptr nocapture %A, i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %3 = load i32, ptr %2, align 4
  %4 = add nsw i32 %3, 6
  store i32 %4, ptr %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}

; Also unroll if we need a runtime check but it was going to be added for
; vectorization anyways.
; CHECK-VECTOR-LABEL: @runtime_chk(
; CHECK-VECTOR: store <4 x float>
; CHECK-VECTOR: store <4 x float>
;
; But not if the unrolling would introduce the runtime check.
; CHECK-SCALAR-LABEL: @runtime_chk(
; CHECK-SCALAR: store float
; CHECK-SCALAR-NOT: store float
define void @runtime_chk(ptr %A, ptr %B, float %N) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %B, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul float %0, %N
  %arrayidx2 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  store float %mul, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
