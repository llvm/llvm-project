; RUN: opt -licm -S %s | FileCheck %s

@A = common global [1024 x float] zeroinitializer, align 4

define i32 @main() nounwind {
entry:
  br label %for.cond

for.cond:
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr [1024 x float]* @A, i64 0, i64 3
  %vecidx = bitcast float* %arrayidx to <4 x float>*
  store <4 x float> zeroinitializer, <4 x float>* %vecidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:
  br label %for.cond

for.end:
  ret i32 0
}

;CHECK: store <4 x float> {{.*}} align 4

