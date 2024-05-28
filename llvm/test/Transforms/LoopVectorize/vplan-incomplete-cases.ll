; REQUIRES: asserts
; RUN: not --crash opt %s -passes=loop-vectorize -S

define void @vplan_incomplete_cases_tc3(i8 %x, i8 %y) {
entry:
  br label %loop.header

loop.header:                                        ; preds = %latch, %entry
  %iv = phi i8 [ %iv.next, %latch ], [ 0, %entry ]
  %and = and i8 %x, %y
  %extract.t = trunc i8 %and to i1
  br i1 %extract.t, label %latch, label %indirect.latch

indirect.latch:                                     ; preds = %loop.header
  br label %latch

latch:                                              ; preds = %indirect.latch, loop.header
  %iv.next = add i8 %iv, 1
  %zext = zext i8 %iv to i32
  %cmp = icmp ult i32 %zext, 2
  br i1 %cmp, label %loop.header, label %exit

exit:                                               ; preds = %latch
  ret void
}
