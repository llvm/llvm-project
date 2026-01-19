; RUN: llc -O2 -march=x86-64 -o /dev/null %s
; Check that CodeGenPrepare does not hang on this input.
; This was caused by an infinite loop between OptimizeNoopCopyExpression
; and optimizePhiType when handling same-type bitcasts.

define void @foo(ptr %p, i1 %cond) {
entry:
  %val = load i32, ptr %p
  br i1 %cond, label %bb1, label %bb2

bb1:
  %c1 = bitcast i32 %val to i32
  br label %bb3

bb2:
  %c2 = bitcast i32 %val to i32
  %c3 = bitcast i32 %c2 to i32
  br label %bb3

bb3:
  %phi = phi i32 [%c3, %bb2], [%c1, %bb1]
  %c4 = bitcast i32 %phi to i32
  store i32 %c4, ptr %p
  ret void
}
