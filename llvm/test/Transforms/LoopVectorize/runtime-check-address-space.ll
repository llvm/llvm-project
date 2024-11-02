; RUN: opt -S -march=r600 -mcpu=cayman -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine < %s | FileCheck %s

; Check vectorization that would ordinarily require a runtime bounds
; check on the pointers when mixing address spaces. For now we cannot
; assume address spaces do not alias, and we can't assume that
; different pointers are directly comparable.
;
; These all test this basic loop for different combinations of address
; spaces, and swapping in globals or adding noalias.
;
;void foo(int addrspace(N)* [noalias] a, int addrspace(M)* [noalias] b, int n)
;{
;    for (int i = 0; i < n; ++i)
;    {
;        a[i] = 3 * b[i];
;    }
;}

; Artificial datalayout
target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"


@g_as1 = common addrspace(1) global [1024 x i32] zeroinitializer, align 16
@q_as2 = common addrspace(2) global [1024 x i32] zeroinitializer, align 16

; Both parameters are unidentified objects with the same address
; space, so this should vectorize normally.
define void @foo(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %n) #0 {
; CHECK-LABEL: @foo(
; CHECK: <4 x i32>
; CHECK: ret

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %idxprom
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %mul = mul nsw i32 %0, 3
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %idxprom1
  store i32 %mul, i32 addrspace(1)* %arrayidx2, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Parameters are unidentified and different address spaces, so cannot vectorize.
define void @bar0(i32* %a, i32 addrspace(1)* %b, i32 %n) #0 {
; CHECK-LABEL: @bar0(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %idxprom
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %mul = mul nsw i32 %0, 3
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %idxprom1
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Swapped arguments should be the same
define void @bar1(i32 addrspace(1)* %a, i32* %b, i32 %n) #0 {
; CHECK-LABEL: @bar1(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %mul = mul nsw i32 %0, 3
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %idxprom1
  store i32 %mul, i32 addrspace(1)* %arrayidx2, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; We should still be able to vectorize with noalias even if the
; address spaces are different.
define void @bar2(i32* noalias %a, i32 addrspace(1)* noalias %b, i32 %n) #0 {
; CHECK-LABEL: @bar2(
; CHECK: <4 x i32>
; CHECK: ret

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %idxprom
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %mul = mul nsw i32 %0, 3
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %idxprom1
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Store to identified global with different address space. This isn't
; generally safe and shouldn't be vectorized.
define void @arst0(i32* %b, i32 %n) #0 {
; CHECK-LABEL: @arst0(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %mul = mul nsw i32 %0, 3
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(1)* @g_as1, i64 0, i64 %idxprom1
  store i32 %mul, i32 addrspace(1)* %arrayidx2, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


; Load from identified global with different address space.
; This isn't generally safe and shouldn't be vectorized.
define void @arst1(i32* %b, i32 %n) #0 {
; CHECK-LABEL: @arst1(
; CHECK-NOT: <4 x i32>
; CHECK: ret

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(1)* @g_as1, i64 0, i64 %idxprom
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %mul = mul nsw i32 %0, 3
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %idxprom1
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Read and write to 2 identified globals in different address
; spaces. This should be vectorized.
define void @aoeu(i32 %n) #0 {
; CHECK-LABEL: @aoeu(
; CHECK: <4 x i32>
; CHECK: ret

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(2)* @q_as2, i64 0, i64 %idxprom
  %0 = load i32, i32 addrspace(2)* %arrayidx, align 4
  %mul = mul nsw i32 %0, 3
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace(1)* @g_as1, i64 0, i64 %idxprom1
  store i32 %mul, i32 addrspace(1)* %arrayidx2, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
