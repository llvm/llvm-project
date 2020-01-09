; RUN: opt < %s -S -analyze -scalar-evolution | FileCheck %s

; copied from flags-from-poison.ll
; CHECK-LABEL: @test-add-nuw
; CHECK: -->  {(1 + %offset)<nuw>,+,1}<nuw><%loop> U: full-set S: full-set
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
; CHECK: -->  {(1 + (10 smax %offset))<nuw>,+,1}<nuw><%loop> U: full-set S: full-set 
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

; CHECK-LABEL: @test-addrec-nsw
; CHECK: -->  {(-1 + (-10 smin %offset))<nsw>,+,-1}<nsw><%loop> U: [-2147483648,1) S: [-2147483648,1)
define void @test-addrec-nsw(float* %input, i32 %offset, i32 %numIterations) {
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
