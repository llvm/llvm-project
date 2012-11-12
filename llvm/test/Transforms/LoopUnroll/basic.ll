; RUN: opt < %s -loop-unroll -S | FileCheck %s


; This should not unroll since the address of the loop header is taken.

; CHECK: @test1
; CHECK: store i8* blockaddress(@test1, %l1), i8** %P
; CHECK: l1:
; CHECK-NEXT: phi i32
; rdar://8287027
define i32 @test1(i8** %P) nounwind ssp {
entry:
  store i8* blockaddress(@test1, %l1), i8** %P
  br label %l1

l1:                                               ; preds = %l1, %entry
  %x.0 = phi i32 [ 0, %entry ], [ %inc, %l1 ]
  %inc = add nsw i32 %x.0, 1
  %exitcond = icmp eq i32 %inc, 3
  br i1 %exitcond, label %l2, label %l1

l2:                                               ; preds = %l1
  ret i32 0
}
