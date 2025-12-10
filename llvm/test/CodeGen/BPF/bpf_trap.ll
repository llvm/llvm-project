; RUN: llc < %s | FileCheck %s
;
target triple = "bpf"

define i32 @test(i8 %x) {
entry:
  %0 = and i8 %x, 3
  switch i8 %0, label %default.unreachable4 [
    i8 0, label %return
    i8 1, label %sw.bb1
    i8 2, label %sw.bb2
    i8 3, label %sw.bb3
  ]

sw.bb1:                                           ; preds = %entry
  br label %return

sw.bb2:                                           ; preds = %entry
  br label %return

sw.bb3:                                           ; preds = %entry
  br label %return

default.unreachable4:                             ; preds = %entry
  unreachable

return:                                           ; preds = %entry, %sw.bb3, %sw.bb2, %sw.bb1
  %retval.0 = phi i32 [ 12, %sw.bb1 ], [ 43, %sw.bb2 ], [ 54, %sw.bb3 ], [ 32, %entry ]
  ret i32 %retval.0
}

; CHECK-NOT: __bpf_trap
