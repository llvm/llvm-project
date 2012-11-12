; RUN: opt < %s -tailcallelim -S | FileCheck %s
; PR7328
; PR7506
define i32 @foo(i32 %x) {
; CHECK: define i32 @foo
; CHECK: %accumulator.tr = phi i32 [ 1, %entry ], [ 0, %body ]
entry:
  %cond = icmp ugt i32 %x, 0                      ; <i1> [#uses=1]
  br i1 %cond, label %return, label %body

body:                                             ; preds = %entry
  %y = add i32 %x, 1                              ; <i32> [#uses=1]
  %tmp = call i32 @foo(i32 %y)                    ; <i32> [#uses=0]
; CHECK-NOT: call
  ret i32 0
; CHECK: ret i32 %accumulator.tr

return:                                           ; preds = %entry
  ret i32 1
}
