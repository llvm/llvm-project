; RUN: llc < %s -mtriple=x86_64-apple-darwin10
; This formerly got DagCombine into a loop, PR 8916.

define i32 @foo(i64 %x, i64 %y, i64 %z, i32 %a, i32 %b) {
entry:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %t1 = shl i64 %x, 15
  %t2 = and i64 %t1, 4294934528
  %t3 = or i64 %t2, %y
  %t4 = xor i64 %z, %t3
  %t5 = trunc i64 %t4 to i32
  %t6 = add i32 %a, %t5
  %t7 = add i32 %t6, %b
  ret i32 %t7
}
