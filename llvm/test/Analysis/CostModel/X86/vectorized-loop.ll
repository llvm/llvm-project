; RUN: opt < %s  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @foo(i32* noalias nocapture %A, i32* noalias nocapture %B, i32 %start, i32 %end) nounwind uwtable ssp {
entry:
  ;CHECK: cost of 1 {{.*}} icmp
  %cmp7 = icmp slt i32 %start, %end
  br i1 %cmp7, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  ;CHECK: cost of 1 {{.*}} sext
  %0 = sext i32 %start to i64
  %1 = sub i32 %end, %start
  %2 = zext i32 %1 to i64
  %end.idx = add i64 %2, %0
  ;CHECK: cost of 1 {{.*}} add
  %n.vec = and i64 %2, 4294967288
  %end.idx.rnd.down = add i64 %n.vec, %0
  ;CHECK: cost of 1 {{.*}} icmp
  %cmp.zero = icmp eq i64 %n.vec, 0
  br i1 %cmp.zero, label %middle.block, label %vector.body

vector.body:                                      ; preds = %for.body.lr.ph, %vector.body
  %index = phi i64 [ %index.next, %vector.body ], [ %0, %for.body.lr.ph ]
  %3 = add i64 %index, 2
  %4 = getelementptr inbounds i32, i32* %B, i64 %3
  ;CHECK: cost of 0 {{.*}} bitcast
  %5 = bitcast i32* %4 to <8 x i32>*
  ;CHECK: cost of 2 {{.*}} load
  %6 = load <8 x i32>, <8 x i32>* %5, align 4
  ;CHECK: cost of 5 {{.*}} mul
  %7 = mul nsw <8 x i32> %6, <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  %8 = getelementptr inbounds i32, i32* %A, i64 %index
  %9 = bitcast i32* %8 to <8 x i32>*
  ;CHECK: cost of 2 {{.*}} load
  %10 = load <8 x i32>, <8 x i32>* %9, align 4
  ;CHECK: cost of 4 {{.*}} add
  %11 = add nsw <8 x i32> %10, %7
  ;CHECK: cost of 2 {{.*}} store
  store <8 x i32> %11, <8 x i32>* %9, align 4
  %index.next = add i64 %index, 8
  %12 = icmp eq i64 %index.next, %end.idx.rnd.down
  ;CHECK: cost of 0 {{.*}} br
  br i1 %12, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body, %for.body.lr.ph
  %cmp.n = icmp eq i64 %end.idx, %end.idx.rnd.down
  br i1 %cmp.n, label %for.end, label %for.body

for.body:                                         ; preds = %middle.block, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %end.idx.rnd.down, %middle.block ]
  %13 = add nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %13
  ;CHECK: cost of 1 {{.*}} load
  %14 = load i32, i32* %arrayidx, align 4
  ;CHECK: cost of 1 {{.*}} mul
  %mul = mul nsw i32 %14, 5
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  ;CHECK: cost of 1 {{.*}} load
  %15 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %15, %mul
  store i32 %add3, i32* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  ;CHECK: cost of 0 {{.*}} trunc
  %16 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %16, %end
  ;CHECK: cost of 0 {{.*}} br
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %middle.block, %for.body, %entry
  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}
