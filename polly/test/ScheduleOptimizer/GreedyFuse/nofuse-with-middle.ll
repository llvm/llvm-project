; RUN: opt %loadPolly -polly-reschedule=0 -polly-loopfusion-greedy=1 -polly-postopts=0 -polly-opt-isl -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-reschedule=1 -polly-loopfusion-greedy=1 -polly-postopts=0 -polly-opt-isl -analyze < %s | FileCheck %s

define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B, i32 %k) {
entry:
  br label %for1


for1:
  %j1 = phi i32 [0, %entry], [%j1.inc, %inc1]
  %j1.cmp = icmp slt i32 %j1, %n
  br i1 %j1.cmp, label %body1, label %exit1

    body1:
      %idx1 = add i32 %j1, %k
      %arrayidx1 = getelementptr inbounds double, double* %A, i32 %j1
      store double 21.0, double* %arrayidx1
      br label %inc1

inc1:
  %j1.inc = add nuw nsw i32 %j1, 1
  br label %for1

exit1:
  br label %middle2


middle2:
  store double 52.0, double* %A
  br label %for3


for3:
  %j3 = phi i32 [0, %middle2], [%j3.inc, %inc3]
  %j3.cmp = icmp slt i32 %j3, %n
  br i1 %j3.cmp, label %body3, label %exit3

    body3:
      %arrayidx3 = getelementptr inbounds double, double* %B, i32 %j3
      store double 84.0, double* %arrayidx3
      br label %inc3

inc3:
  %j3.inc = add nuw nsw i32 %j3, 1
  br label %for3

exit3:
  br label %return


return:
  ret void
}


; CHECK:      Calculated schedule:
; CHECK-NEXT:   n/a
