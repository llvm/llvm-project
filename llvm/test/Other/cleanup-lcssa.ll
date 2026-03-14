; RUN: opt -S -O3 < %s | FileCheck %s

define i64 @test() {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i64 %i, 1
  %cond = tail call i1 @check()
  br i1 %cond, label %loop, label %exit

exit:
  ; CHECK-NOT: lcssa
  ret i64 %i
}

declare i1 @check()
