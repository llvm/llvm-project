; RUN: opt -passes=unify-loop-exits -S %s
; Based from this issue: https://github.com/llvm/llvm-project/issues/165252

define void @test() {
entry:
  br i1 true, label %end, label %Loop

Loop:
  %V = phi i32 [0, %entry], [%V1, %Loop]
  %V1 = add i32 %V, 1
  br label %Loop

end:
  ret void
}
