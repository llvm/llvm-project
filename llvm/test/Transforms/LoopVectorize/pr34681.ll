; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check the scenario where we have an unknown Stride, which happens to also be
; the loop iteration count, so if we specialize the loop for the Stride==1 case,
; this also implies that the loop will iterate no more than a single iteration,
; as in the following example: 
;
;       unsigned int N;
;       int tmp = 0;
;       for(unsigned int k=0;k<N;k++) {
;         tmp+=(int)B[k*N+j];
;       }
;
; We check here that the following runtime scev guard for Stride==1 is NOT generated:
; vector.scevcheck:
;   %ident.check = icmp ne i32 %N, 1
;   %0 = or i1 false, %ident.check
;   br i1 %0, label %scalar.ph, label %vector.ph
; Instead the loop is vectorized with an unknown stride.

; CHECK-LABEL: @foo1
; CHECK: for.body.lr.ph
; CHECK-NOT: %ident.check = icmp ne i32 %N, 1
; CHECK-NOT: %[[TEST:[0-9]+]] = or i1 false, %ident.check
; CHECK-NOT: br i1 %[[TEST]], label %scalar.ph, label %vector.ph
; CHECK: vector.ph
; CHECK: vector.body
; CHECK: <4 x i32>
; CHECK: middle.block
; CHECK: scalar.ph


define i32 @foo1(i32 %N, i16* nocapture readnone %A, i16* nocapture readonly %B, i32 %i, i32 %j)  {
entry:
  %cmp8 = icmp eq i32 %N, 0
  br i1 %cmp8, label %for.end, label %for.body.lr.ph

for.body.lr.ph:
  br label %for.body

for.body:
  %tmp.010 = phi i32 [ 0, %for.body.lr.ph ], [ %add1, %for.body ]
  %k.09 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %mul = mul i32 %k.09, %N
  %add = add i32 %mul, %j
  %arrayidx = getelementptr inbounds i16, i16* %B, i32 %add
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  %add1 = add nsw i32 %tmp.010, %conv
  %inc = add nuw i32 %k.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  %add1.lcssa = phi i32 [ %add1, %for.body ]
  br label %for.end

for.end: 
  %tmp.0.lcssa = phi i32 [ 0, %entry ], [ %add1.lcssa, %for.end.loopexit ]
  ret i32 %tmp.0.lcssa
}


; Check the same, but also where the Stride and the loop iteration count
; are not of the same data type. 
;
;       unsigned short N;
;       int tmp = 0;
;       for(unsigned int k=0;k<N;k++) {
;         tmp+=(int)B[k*N+j];
;       }
;
; We check here that the following runtime scev guard for Stride==1 is NOT generated:
; vector.scevcheck:
; %ident.check = icmp ne i16 %N, 1
; %0 = or i1 false, %ident.check
; br i1 %0, label %scalar.ph, label %vector.ph


; CHECK-LABEL: @foo2
; CHECK: for.body.lr.ph
; CHECK-NOT: %ident.check = icmp ne i16 %N, 1
; CHECK-NOT: %[[TEST:[0-9]+]] = or i1 false, %ident.check
; CHECK-NOT: br i1 %[[TEST]], label %scalar.ph, label %vector.ph
; CHECK: vector.ph
; CHECK: vector.body
; CHECK: <4 x i32>
; CHECK: middle.block
; CHECK: scalar.ph

define i32 @foo2(i16 zeroext %N, i16* nocapture readnone %A, i16* nocapture readonly %B, i32 %i, i32 %j) {
entry:
  %conv = zext i16 %N to i32
  %cmp11 = icmp eq i16 %N, 0
  br i1 %cmp11, label %for.end, label %for.body.lr.ph

for.body.lr.ph:
  br label %for.body

for.body:
  %tmp.013 = phi i32 [ 0, %for.body.lr.ph ], [ %add4, %for.body ]
  %k.012 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %mul = mul nuw i32 %k.012, %conv
  %add = add i32 %mul, %j
  %arrayidx = getelementptr inbounds i16, i16* %B, i32 %add
  %0 = load i16, i16* %arrayidx, align 2
  %conv3 = sext i16 %0 to i32
  %add4 = add nsw i32 %tmp.013, %conv3
  %inc = add nuw nsw i32 %k.012, 1
  %exitcond = icmp eq i32 %inc, %conv
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:
  %add4.lcssa = phi i32 [ %add4, %for.body ]
  br label %for.end

for.end:
  %tmp.0.lcssa = phi i32 [ 0, %entry ], [ %add4.lcssa, %for.end.loopexit ]
  ret i32 %tmp.0.lcssa
}
