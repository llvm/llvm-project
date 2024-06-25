; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Check IndexedReference::computeRefCost can handle type differences between
; Stride and TripCount

; CHECK: Loop 'for.cond' has cost = 64

%struct._Handleitem = type { ptr }

define void @handle_to_ptr(ptr %blocks) {
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
  %arrayidx = getelementptr inbounds ptr, ptr %blocks, i64 %idxprom
  store ptr null, ptr %arrayidx, align 8
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

; Exit blocks
for.end:                                          ; preds = %for.cond
  ret void
}

; Check IndexedReference::computeRefCost can handle type differences between
; Coeff and ElemSize.

; CHECK: Loop 'for.cond' has cost = 100000000
; CHECK: Loop 'for.cond1' has cost = 1000000
; CHECK: Loop 'for.cond5' has cost = 40000

@data = external dso_local global [2 x [4 x [18 x i32]]], align 1

define dso_local void @handle_to_ptr_2(i1 %b0, i1 %b1, i1 %b2) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i16 [ 0, %entry ], [ %inc18, %for.inc17 ]
  %idxprom = zext i16 %i.0 to i32
  br i1 %b2, label %for.end19, label %for.cond1

for.cond1:
  %j.0 = phi i16 [ %inc15, %for.inc14 ], [ 0, %for.cond ]
  br i1 %b1, label %for.inc17, label %for.cond5.preheader

for.cond5.preheader:
  %idxprom10 = zext i16 %j.0 to i32
  br label %for.cond5

for.cond5:
  %k.0 = phi i16 [ %inc, %for.inc ], [ 0, %for.cond5.preheader ]
  br i1 %b0, label %for.inc14, label %for.inc

for.inc:
  %idxprom12 = zext i16 %k.0 to i32
  %arrayidx13 = getelementptr inbounds [2 x [4 x [18 x i32]]], ptr @data, i32 0, i32 %idxprom, i32 %idxprom10, i32 %idxprom12
  store i32 7, ptr %arrayidx13, align 1
  %inc = add nuw nsw i16 %k.0, 1
  br label %for.cond5

for.inc14:
  %inc15 = add nuw nsw i16 %j.0, 1
  br label %for.cond1

for.inc17:
  %inc18 = add nuw nsw i16 %i.0, 1
  br label %for.cond

for.end19:
  ret void
}

; Check IndexedReference::computeRefCost can handle negative stride

; CHECK: Loop 'for.neg.cond' has cost = 64

define void @handle_to_ptr_neg_stride(ptr %blocks) {
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
  %arrayidx = getelementptr inbounds ptr, ptr %blocks, i64 %idxprom
  store ptr null, ptr %arrayidx, align 8
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

; CHECK: Loop 'for.cond2' has cost = 2561
define void @Test2(ptr %B) {
entry:
  br label %for.cond2

for.cond2:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 40960, %entry ], [ %dec, %for.body ]
  %cmp = icmp sgt i32 %i.0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = sub nsw i32 40960, %i.0
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds double, ptr %B, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds double, ptr %B, i64 %idxprom1
  store double %0, ptr %arrayidx2, align 8
  %dec = add nsw i32 %i.0, -1
  br label %for.cond2

for.end:                                          ; preds = %for.cond
  ret void
}



;   for (i = 40960; i > 0; i--)
;     C[i] = C[i];

; CHECK: Loop 'for.cond3' has cost = 2561
define void @Test3(ptr %C) {
entry:
  br label %for.cond3

for.cond3:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 40960, %entry ], [ %dec, %for.body ]
  %cmp = icmp sgt i32 %i.0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds ptr, ptr %C, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds ptr, ptr %C, i64 %idxprom1
  store ptr %0, ptr %arrayidx2, align 8
  %dec = add nsw i32 %i.0, -1
  br label %for.cond3

for.end:                                          ; preds = %for.cond
  ret void
}



;  for (i = 0; i < 40960; i++)
;     D[i] = D[i];

; CHECK: Loop 'for.cond4' has cost = 2561
define void @Test4(ptr %D) {
entry:
  br label %for.cond4

for.cond4:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 40960
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds ptr, ptr %D, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds ptr, ptr %D, i64 %idxprom1
  store ptr %0, ptr %arrayidx2, align 8
  %inc = add nsw i32 %i.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond
  ret void
}
