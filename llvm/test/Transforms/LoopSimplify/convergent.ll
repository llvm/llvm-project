; RUN: opt < %s -S -passes=loop-simplify | FileCheck %s

; Don't separate out nested loops if a convergent call is present

; CHECK-NOT: BB1.outer
; CHECK: BB1.backedge

define i32 @test(i1 %loop_cond, i1 %exit_cond, i32 %init) {
entry:
  br label %BB1

BB1:
  %indvar = phi i32 [%indvar, %BB1], [%inc, %BB2], [%init, %entry]
  call void @f() convergent
  br i1 %loop_cond, label %BB1, label %BB2

BB2:
  %inc = add i32 %indvar, 1
  br i1 %exit_cond, label %exit, label %BB1

exit:
  ret i32 %inc
}

declare void @f() convergent
