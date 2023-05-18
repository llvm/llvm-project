; RUN: llc < %s -mtriple thumbv6-apple-macosx10.6.0 | FileCheck %s

; test that we print the label of a bb that is only used in a jump table.

; CHECK:	.long	[[JUMPTABLE_DEST:LBB[0-9]+_[0-9]+]]
; CHECK: [[JUMPTABLE_DEST]]:

define i32 @calculate()  {
entry:
  switch i32 undef, label %return [
    i32 1, label %sw.bb
    i32 2, label %sw.bb6
    i32 3, label %sw.bb13
    i32 4, label %sw.bb20
  ]

sw.bb:                                            ; preds = %entry
  br label %return

sw.bb6:                                           ; preds = %entry
  br label %return

sw.bb13:                                          ; preds = %entry
  br label %return

sw.bb20:                                          ; preds = %entry
  %div = sdiv i32 undef, undef
  br label %return

return:                                           ; preds = %sw.bb20, %sw.bb13, %sw.bb6, %sw.bb, %entry
  %retval.0 = phi i32 [ %div, %sw.bb20 ], [ undef, %sw.bb13 ], [ undef, %sw.bb6 ], [ undef, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}
