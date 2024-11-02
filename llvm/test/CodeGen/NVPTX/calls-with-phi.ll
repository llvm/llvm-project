; RUN: llc < %s -march=nvptx 2>&1 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx | %ptxas-verify %}

; Make sure the example doesn't crash with segfault

; CHECK: .visible .func ({{.*}}) loop
define i32 @loop(i32, i32) {
entry:
  br label %loop

loop:
  %i = phi i32 [ %0, %entry ], [ %res, %loop ]
  %res = call i32 @div(i32 %i, i32 %1)

  %exitcond = icmp eq i32 %res, %0
  br i1 %exitcond, label %exit, label %loop

exit:
  ret i32 %res
}

define i32 @div(i32, i32) {
  ret i32 0
}
