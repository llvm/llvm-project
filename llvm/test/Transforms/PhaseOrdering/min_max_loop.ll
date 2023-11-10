; RUN: opt < %s --O3 -S | FileCheck %s
; See issue #55013 and PR #70845 for more details.
; This test comes from the following C program, compiled with : 
;   > clang -O0 -S -emit-llvm -fno-discard-value-names
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

define signext i16 @vecreduce_smin_v2i16(i32 noundef %n, ptr noundef %v) {
  ;; CHECK: @llvm.smin
entry:
  %n.addr = alloca i32, align 4
  %v.addr = alloca ptr, align 8
  %p = alloca i16, align 2
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store ptr %v, ptr %v.addr, align 8
  store i16 0, ptr %p, align 2
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %n.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i16, ptr %p, align 2
  %conv = sext i16 %2 to i32
  %3 = load ptr, ptr %v.addr, align 8
  %4 = load i32, ptr %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds i16, ptr %3, i64 %idxprom
  %5 = load i16, ptr %arrayidx, align 2
  %conv1 = sext i16 %5 to i32
  %cmp2 = icmp sgt i32 %conv, %conv1
  br i1 %cmp2, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body
  %6 = load ptr, ptr %v.addr, align 8
  %7 = load i32, ptr %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %arrayidx5 = getelementptr inbounds i16, ptr %6, i64 %idxprom4
  %8 = load i16, ptr %arrayidx5, align 2
  %conv6 = sext i16 %8 to i32
  br label %cond.end

cond.false:                                       ; preds = %for.body
  %9 = load i16, ptr %p, align 2
  %conv7 = sext i16 %9 to i32
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %conv6, %cond.true ], [ %conv7, %cond.false ]
  %conv8 = trunc i32 %cond to i16
  store i16 %conv8, ptr %p, align 2
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %10 = load i32, ptr %i, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %11 = load i16, ptr %p, align 2
  ret i16 %11
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define signext i16 @vecreduce_smax_v2i16(i32 noundef %n, ptr noundef %v) {
  ;; CHECK: @llvm.smax
entry:
  %n.addr = alloca i32, align 4
  %v.addr = alloca ptr, align 8
  %p = alloca i16, align 2
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store ptr %v, ptr %v.addr, align 8
  store i16 0, ptr %p, align 2
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %n.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i16, ptr %p, align 2
  %conv = sext i16 %2 to i32
  %3 = load ptr, ptr %v.addr, align 8
  %4 = load i32, ptr %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds i16, ptr %3, i64 %idxprom
  %5 = load i16, ptr %arrayidx, align 2
  %conv1 = sext i16 %5 to i32
  %cmp2 = icmp slt i32 %conv, %conv1
  br i1 %cmp2, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body
  %6 = load ptr, ptr %v.addr, align 8
  %7 = load i32, ptr %i, align 4
  %idxprom4 = sext i32 %7 to i64
  %arrayidx5 = getelementptr inbounds i16, ptr %6, i64 %idxprom4
  %8 = load i16, ptr %arrayidx5, align 2
  %conv6 = sext i16 %8 to i32
  br label %cond.end

cond.false:                                       ; preds = %for.body
  %9 = load i16, ptr %p, align 2
  %conv7 = sext i16 %9 to i32
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %conv6, %cond.true ], [ %conv7, %cond.false ]
  %conv8 = trunc i32 %cond to i16
  store i16 %conv8, ptr %p, align 2
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %10 = load i32, ptr %i, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %11 = load i16, ptr %p, align 2
  ret i16 %11
}