; RUN: opt < %s -O3 -S | FileCheck %s
; See issue #55013 and PR #70845 for more details.
; This test comes from the following C program, compiled with clang
;
;; short vecreduce_smin_v2i16(int n, short* v)
;; {
;;   short p = 0;
;;   for (int i = 0; i < n; ++i)
;;     p = p > v[i] ? v[i] : p;
;;   return p;
;; }
;
;; short vecreduce_smax_v2i16(int n, short* v)
;; {
;;   short p = 0;
;;   for (int i = 0; i < n; ++i)
;;     p = p < v[i] ? v[i] : p;
;;   return p;
;; }

define i16 @vecreduce_smin_v2i16(i32 %n, ptr %v) {
; CHECK-LABEL: define i16 @vecreduce_smin_v2i16(
; CHECK:    @llvm.smin.v2i16

entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.0 = phi i16 [ 0, %entry ], [ %conv8, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = sext i16 %p.0 to i32
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i16, ptr %v, i64 %idxprom
  %0 = load i16, ptr %arrayidx, align 2
  %conv1 = sext i16 %0 to i32
  %cmp2 = icmp sgt i32 %conv, %conv1
  br i1 %cmp2, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body
  %idxprom4 = sext i32 %i.0 to i64
  %arrayidx5 = getelementptr inbounds i16, ptr %v, i64 %idxprom4
  %1 = load i16, ptr %arrayidx5, align 2
  %conv6 = sext i16 %1 to i32
  br label %cond.end

cond.false:                                       ; preds = %for.body
  %conv7 = sext i16 %p.0 to i32
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %conv6, %cond.true ], [ %conv7, %cond.false ]
  %conv8 = trunc i32 %cond to i16
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i16 %p.0
}

define i16 @vecreduce_smax_v2i16(i32 %n, ptr %v) {
; CHECK-LABEL: define i16 @vecreduce_smax_v2i16(
; CHECK:  @llvm.smax.v2i16

entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.0 = phi i16 [ 0, %entry ], [ %conv8, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = sext i16 %p.0 to i32
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i16, ptr %v, i64 %idxprom
  %0 = load i16, ptr %arrayidx, align 2
  %conv1 = sext i16 %0 to i32
  %cmp2 = icmp slt i32 %conv, %conv1
  br i1 %cmp2, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body
  %idxprom4 = sext i32 %i.0 to i64
  %arrayidx5 = getelementptr inbounds i16, ptr %v, i64 %idxprom4
  %1 = load i16, ptr %arrayidx5, align 2
  %conv6 = sext i16 %1 to i32
  br label %cond.end

cond.false:                                       ; preds = %for.body
  %conv7 = sext i16 %p.0 to i32
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %conv6, %cond.true ], [ %conv7, %cond.false ]
  %conv8 = trunc i32 %cond to i16
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i16 %p.0
}
