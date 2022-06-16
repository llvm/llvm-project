; RUN: opt < %s  -opaque-pointers -cache-line-size=32 -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck -check-prefix=SMALLER-CACHELINE %s
; RUN: opt < %s  -opaque-pointers -cache-line-size=256 -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck -check-prefix=LARGER-CACHELINE %s

;; This test is similar to test/Analysis/LoopCacheAnalysis/PowerPC/compute-cost.ll,
;; with differences that it tests the scenarios where an option for cache line size is
;; specified with different values.

; Check IndexedReference::computeRefCost can handle type differences between
; Stride and TripCount

; SMALLER-CACHELINE: Loop 'for.cond' has cost = 256
; LARGER-CACHELINE: Loop 'for.cond' has cost = 32
%struct._Handleitem = type { %struct._Handleitem* }

define void @handle_to_ptr(%struct._Handleitem** %blocks) {
; Preheader:
entry:
  br label %for.cond

; Loop:
for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds %struct._Handleitem*, %struct._Handleitem** %blocks, i64 %idxprom
  store %struct._Handleitem* null, %struct._Handleitem** %arrayidx, align 8
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

; Exit blocks
for.end:                                          ; preds = %for.cond
  ret void
}



; Check IndexedReference::computeRefCost can handle negative stride

; SMALLER-CACHELINE: Loop 'for.neg.cond' has cost = 256
; LARGER-CACHELINE: Loop 'for.neg.cond' has cost = 32
define void @handle_to_ptr_neg_stride(%struct._Handleitem** %blocks) {
; Preheader:
entry:
  br label %for.neg.cond

; Loop:
for.neg.cond:                                         ; preds = %for.neg.body, %entry
  %i.0 = phi i32 [ 1023, %entry ], [ %dec, %for.neg.body ]
  %cmp = icmp sgt i32 %i.0, 0
  br i1 %cmp, label %for.neg.body, label %for.neg.end

for.neg.body:                                         ; preds = %for.neg.cond
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds %struct._Handleitem*, %struct._Handleitem** %blocks, i64 %idxprom
  store %struct._Handleitem* null, %struct._Handleitem** %arrayidx, align 8
  %dec = add nsw i32 %i.0, -1
  br label %for.neg.cond

; Exit blocks
for.neg.end:                                          ; preds = %for.neg.cond
  ret void
}



;   for (int i = 40960; i > 0; i--)
;     B[i] = B[40960 - i];

; FIXME: Currently negative access functions are treated the same as positive
; access functions. When this is fixed this testcase should have a cost
; approximately 2x higher.

; SMALLER-CACHELINE: Loop 'for.cond2' has cost = 10240
; LARGER-CACHELINE: Loop 'for.cond2' has cost = 1280
define void @Test2(double* %B) {
entry:
  br label %for.cond2

for.cond2:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 40960, %entry ], [ %dec, %for.body ]
  %cmp = icmp sgt i32 %i.0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = sub nsw i32 40960, %i.0
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds double, double* %B, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds double, double* %B, i64 %idxprom1
  store double %0, double* %arrayidx2, align 8
  %dec = add nsw i32 %i.0, -1
  br label %for.cond2

for.end:                                          ; preds = %for.cond
  ret void
}



;   for (i = 40960; i > 0; i--)
;     C[i] = C[i];

; SMALLER-CACHELINE: Loop 'for.cond3' has cost = 10240
; LARGER-CACHELINE: Loop 'for.cond3' has cost = 1280
define void @Test3(double** %C) {
entry:
  br label %for.cond3

for.cond3:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 40960, %entry ], [ %dec, %for.body ]
  %cmp = icmp sgt i32 %i.0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds double*, double** %C, i64 %idxprom
  %0 = load double*, double** %arrayidx, align 8
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds double*, double** %C, i64 %idxprom1
  store double* %0, double** %arrayidx2, align 8
  %dec = add nsw i32 %i.0, -1
  br label %for.cond3

for.end:                                          ; preds = %for.cond
  ret void
}



;  for (i = 0; i < 40960; i++)
;     D[i] = D[i];

; SMALLER-CACHELINE: Loop 'for.cond4' has cost = 10240
; LARGER-CACHELINE: Loop 'for.cond4' has cost = 1280
define void @Test4(double** %D) {
entry:
  br label %for.cond4

for.cond4:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 40960
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds double*, double** %D, i64 %idxprom
  %0 = load double*, double** %arrayidx, align 8
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds double*, double** %D, i64 %idxprom1
  store double* %0, double** %arrayidx2, align 8
  %inc = add nsw i32 %i.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond
  ret void
}
