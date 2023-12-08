; RUN: llc < %s
target triple = "x86_64-unknown-linux-gnu"

@g = external global [10 x i32]

define void @f(i32 %u) {
  %1 = add i32 %u, 4
  br label %P.Proc8.exit

P.Proc8.exit:
  %valueindex35.i = getelementptr [10 x i32], ptr @g, i32 0, i32 %1
  store i32 %u, ptr %valueindex35.i
  ret void
}
