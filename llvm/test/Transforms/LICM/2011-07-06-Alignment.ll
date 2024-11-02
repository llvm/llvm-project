; RUN: opt -licm -S < %s | FileCheck %s

@A = common global [1024 x float] zeroinitializer, align 4

define i32 @main() nounwind {
entry:
  br label %for.cond

for.cond:
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr [1024 x float], ptr @A, i64 0, i64 3
  store <4 x float> zeroinitializer, ptr %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:
  br label %for.cond

for.end:
  ret i32 0
}

;CHECK: store <4 x float> {{.*}} align 4

