; RUN: opt < %s -S -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -S -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; copied from flags-from-poison.ll
; CHECK-LABEL: @test-add-nuw
; CHECK: -->  {(1 + %offset)<nuw>,+,1}<nuw><%loop> U: [1,0) S: [1,0)
define void @test-add-nuw(float* %input, i32 %offset, i32 %numIterations) {
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %nexti = add nuw i32 %i, 1
  %index32 = add nuw i32 %nexti, %offset
  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: @test-addrec-nuw
; CHECK: -->  {(1 + (10 smax %offset))<nuw>,+,1}<nuw><%loop> U: [11,0) S: [11,0)
define void @test-addrec-nuw(float* %input, i32 %offset, i32 %numIterations) {
entry:
  %cmp = icmp sgt i32 %offset, 10
  %min.10 = select i1 %cmp, i32 %offset, i32 10
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %nexti = add nuw i32 %i, 1
  %index32 = add nuw i32 %nexti, %min.10
  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: @test-addrec-nsw-start-neg-strip-neg
; CHECK: -->  {(-1 + (-10 smin %offset))<nsw>,+,-1}<nsw><%loop> U: [-2147483648,-10) S: [-2147483648,-10)
define void @test-addrec-nsw-start-neg-strip-neg(float* %input, i32 %offset, i32 %numIterations) {
entry:
  %cmp = icmp slt i32 %offset, -10
  %max = select i1 %cmp, i32 %offset, i32 -10
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %nexti = add nsw i32 %i, -1
  %index32 = add nsw i32 %nexti, %max
  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: @test-addrec-nsw-start-pos-strip-neg
; CHECK: -->  {(-1 + (10 smin %offset))<nsw>,+,-1}<nsw><%loop> U: [-2147483648,10) S: [-2147483648,10)
define void @test-addrec-nsw-start-pos-strip-neg(float* %input, i32 %offset, i32 %numIterations) {
entry:
  %cmp = icmp slt i32 %offset, 10
  %max = select i1 %cmp, i32 %offset, i32  10
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %nexti = add nsw i32 %i, -1
  %index32 = add nsw i32 %nexti, %max
  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: @test-addrec-nsw-start-pos-strip-pos
; CHECK: -->  {(1 + (10 smax %offset))<nuw><nsw>,+,1}<nuw><nsw><%loop> U: [11,-2147483648) S: [11,-2147483648)
define void @test-addrec-nsw-start-pos-strip-pos(float* %input, i32 %offset, i32 %numIterations) {
entry:
  %cmp = icmp sgt i32 %offset, 10
  %min = select i1 %cmp, i32 %offset, i32  10
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %nexti = add nsw i32 %i, 1
  %index32 = add nsw i32 %nexti, %min
  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: @test-addrec-nsw-start-neg-strip-pos
; CHECK: -->  {(1 + (-10 smax %offset))<nsw>,+,1}<nsw><%loop> U: [-9,-2147483648) S: [-9,-2147483648)
define void @test-addrec-nsw-start-neg-strip-pos(float* %input, i32 %offset, i32 %numIterations) {
entry:
  %cmp = icmp sgt i32 %offset, -10
  %min = select i1 %cmp, i32 %offset, i32  -10
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]
  %nexti = add nsw i32 %i, 1
  %index32 = add nsw i32 %nexti, %min
  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

