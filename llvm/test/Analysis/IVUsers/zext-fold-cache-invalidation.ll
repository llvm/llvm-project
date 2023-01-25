; RUN: opt -verify-scev -passes='print<iv-users>' -disable-output %s 2>&1 | FileCheck %s

target datalayout = "n16"

define i16 @zext_cache_invalidation_1(i1 %c) {
; CHECK:      IV Users for loop %loop with backedge-taken count 13:
; CHECK-NEXT:   %iv = {-3,+,4}<nuw><nsw><%loop> in    %iv.ext = zext i16 %iv to i32
;
entry:
  br i1 false, label %loop, label %exit

loop:
  %iv = phi i16 [ -3, %entry ], [ %iv.next, %loop ]
  %iv.ext = zext i16 %iv to i32
  %iv.inc = add i32 %iv.ext, 4
  %iv.next = trunc i32 %iv.inc to i16
  %cond = icmp ult i16 %iv.next, 51
  br i1 %cond, label %loop, label %exit

exit:
  ret i16 0
}
