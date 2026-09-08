; RUN: opt -disable-output -passes='print<must-execute>' %s 2>&1 | FileCheck %s

; The loop is not in simplified form: %header is entered from %entry, which
; ends in a conditional branch, so the loop has no preheader. %entry is still
; the only predecessor of the header from outside the loop, so the value %iv
; starts at is known, and %latch is proven to execute on the first iteration.

; CHECK-LABEL: @no_preheader(
; CHECK:       header:
; CHECK-NEXT:    %iv = phi i32 [ 0, %latch ], [ 0, %entry ] ; (mustexec in: header)
; CHECK-NEXT:    %cmp = icmp sgt i32 %iv, 0 ; (mustexec in: header)
; CHECK-NEXT:    br i1 %cmp, label %loop.exit, label %latch ; (mustexec in: header)
; CHECK:       latch:
; CHECK-NEXT:    br label %header ; (mustexec in: header)

define i32 @no_preheader(i1 %c) {
entry:
  br i1 %c, label %header, label %exit

header:
  %iv = phi i32 [ 0, %latch ], [ 0, %entry ]
  %cmp = icmp sgt i32 %iv, 0
  br i1 %cmp, label %loop.exit, label %latch

latch:
  br label %header

loop.exit:
  ret i32 0

exit:
  ret i32 0
}
