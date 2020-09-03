; RUN: opt -basic-aa -loop-vectorize -enable-mem-access-versioning -force-vector-width=2 -force-vector-interleave=1 < %s -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Check that we version this loop with speculating the value 1 for symbolic
; strides.  This also checks that the symbolic stride information is correctly
; propagated to the memcheck generation.  Without this the loop wouldn't
; vectorize because we couldn't determine the array bounds for the required
; memchecks.

; CHECK-LABEL: test
define void @test(i32*  %A, i64 %AStride,
                  i32*  %B, i32 %BStride,
                  i32*  %C, i64 %CStride, i32 %N) {
entry:
  %cmp13 = icmp eq i32 %N, 0
  br i1 %cmp13, label %for.end, label %for.body.preheader

; CHECK-DAG: icmp ne i64 %AStride, 1
; CHECK-DAG: icmp ne i32 %BStride, 1
; CHECK-DAG: icmp ne i64 %CStride, 1
; CHECK: or
; CHECK: or
; CHECK: br

; CHECK: vector.body
; CHECK: load <2 x i32>

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %iv.trunc = trunc i64 %indvars.iv to i32
  %mul = mul i32 %iv.trunc, %BStride
  %mul64 = zext i32 %mul to i64
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %mul64
  %0 = load i32, i32* %arrayidx, align 4
  %mul2 = mul nsw i64 %indvars.iv, %CStride
  %arrayidx3 = getelementptr inbounds i32, i32* %C, i64 %mul2
  %1 = load i32, i32* %arrayidx3, align 4
  %mul4 = mul nsw i32 %1, %0
  %mul3 = mul nsw i64 %indvars.iv, %AStride
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %mul3
  store i32 %mul4, i32* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; We used to crash on this function because we removed the fptosi cast when
; replacing the symbolic stride '%conv'.
; PR18480

; CHECK-LABEL: fn1
; CHECK: load <2 x double>

define void @fn1(double* noalias %x, double* noalias %c, double %a) {
entry:
  %conv = fptosi double %a to i32
  %conv2 = add i32 %conv, 4
  %cmp8 = icmp sgt i32 %conv2, 0
  br i1 %cmp8, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %0 = trunc i64 %indvars.iv to i32
  %mul = mul nsw i32 %0, %conv
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds double, double* %x, i64 %idxprom
  %1 = load double, double* %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds double, double* %c, i64 %indvars.iv
  store double %1, double* %arrayidx3, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %conv2
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
