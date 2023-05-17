; RUN: opt < %s -passes=inline -inline-threshold=20 -S -mtriple=aarch64-none-linux  | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -inline-threshold=20 -S -mtriple=aarch64-none-linux | FileCheck %s

define i32 @callee_range(i32 %a, ptr %P) {
  switch i32 %a, label %sw.default [
    i32 0, label %sw.bb0
    i32 1000, label %sw.bb1
    i32 2000, label %sw.bb1
    i32 3000, label %sw.bb1
    i32 4000, label %sw.bb1
    i32 5000, label %sw.bb1
    i32 6000, label %sw.bb1
    i32 7000, label %sw.bb1
    i32 8000, label %sw.bb1
    i32 9000, label %sw.bb1
  ]

sw.default:
  store volatile i32 %a, ptr %P
  br label %return
sw.bb0:
  store volatile i32 %a, ptr %P
  br label %return
sw.bb1:
  store volatile i32 %a, ptr %P
  br label %return
return:
  ret i32 42
}

define i32 @caller_range(i32 %a, ptr %P) {
; CHECK-LABEL: @caller_range(
; CHECK: call i32 @callee_range
  %r = call i32 @callee_range(i32 %a, ptr %P)
  ret i32 %r
}

define i32 @callee_bittest(i32 %a, ptr %P) {
  switch i32 %a, label %sw.default [
    i32 0, label %sw.bb0
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb0
    i32 4, label %sw.bb1
    i32 5, label %sw.bb2
    i32 6, label %sw.bb0
    i32 7, label %sw.bb1
    i32 8, label %sw.bb2
  ]

sw.default:
  store volatile i32 %a, ptr %P
  br label %return

sw.bb0:
  store volatile i32 %a, ptr %P
  br label %return

sw.bb1:
  store volatile i32 %a, ptr %P
  br label %return

sw.bb2:
  br label %return

return:
  ret i32 42
}


define i32 @caller_bittest(i32 %a, ptr %P) {
; CHECK-LABEL: @caller_bittest(
; CHECK-NOT: call i32 @callee_bittest
  %r= call i32 @callee_bittest(i32 %a, ptr %P)
  ret i32 %r
}

define i32 @callee_jumptable(i32 %a, ptr %P) {
  switch i32 %a, label %sw.default [
    i32 1001, label %sw.bb101
    i32 1002, label %sw.bb102
    i32 1003, label %sw.bb103
    i32 1004, label %sw.bb104
    i32 1005, label %sw.bb101
    i32 1006, label %sw.bb102
    i32 1007, label %sw.bb103
    i32 1008, label %sw.bb104
    i32 1009, label %sw.bb101
    i32 1010, label %sw.bb102
    i32 1011, label %sw.bb103
    i32 1012, label %sw.bb104
 ]

sw.default:
  br label %return

sw.bb101:
  store volatile i32 %a, ptr %P
  br label %return

sw.bb102:
  store volatile i32 %a, ptr %P
  br label %return

sw.bb103:
  store volatile i32 %a, ptr %P
  br label %return

sw.bb104:
  store volatile i32 %a, ptr %P
  br label %return

return:
  ret i32 42
}

define i32 @caller_jumptable(i32 %a, i32 %b, ptr %P) {
; CHECK-LABEL: @caller_jumptable(
; CHECK: call i32 @callee_jumptable
  %r = call i32 @callee_jumptable(i32 %b, ptr %P)
  ret i32 %r
}


define internal i32 @callee_negativeCost(i32 %t)  {
entry:
  switch i32 %t, label %sw.default [
    i32 1, label %sw.bb
    i32 0, label %sw.bb1
    i32 42, label %sw.bb2
    i32 43, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  br label %cleanup

sw.bb1:                                           ; preds = %entry
  br label %cleanup

sw.bb2:                                           ; preds = %entry
  br label %cleanup

sw.bb3:                                           ; preds = %entry
  br label %cleanup

sw.default:                                       ; preds = %entry
  br label %cleanup

cleanup:                                          ; preds = %sw.default, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %retval.0 = phi i32 [ 1, %sw.default ], [ 3, %sw.bb3 ], [ 2, %sw.bb2 ], [ 0, %sw.bb1 ], [ 0, %sw.bb ]
  ret i32 %retval.0
}

define i32 @caller_negativeCost(i32 %t) {
; CHECK-LABEL: @caller_negativeCost(
; CHECK-NOT: call i32 @callee_negativeCost
entry:
  %call = call i32 @callee_negativeCost(i32 %t)
  ret i32 %call
}
