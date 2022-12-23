; RUN: opt < %s -passes=loop-unroll -S | FileCheck %s
; RUN: opt < %s -passes='require<opt-remark-emit>,loop-unroll' -S | FileCheck %s


; This should not unroll since the address of the loop header is taken.

; CHECK-LABEL: @test1(
; CHECK: store ptr blockaddress(@test1, %l1), ptr %P
; CHECK: l1:
; CHECK-NEXT: phi i32
; rdar://8287027
define i32 @test1(ptr %P) nounwind ssp {
entry:
  store ptr blockaddress(@test1, %l1), ptr %P
  br label %l1

l1:                                               ; preds = %l1, %entry
  %x.0 = phi i32 [ 0, %entry ], [ %inc, %l1 ]
  %inc = add nsw i32 %x.0, 1
  %exitcond = icmp eq i32 %inc, 3
  br i1 %exitcond, label %l2, label %l1

l2:                                               ; preds = %l1
  ret i32 0
}

; This should not unroll since the call is 'noduplicate'.

; CHECK-LABEL: @test2(
define i32 @test2(ptr %P) nounwind ssp {
entry:
  br label %l1

l1:                                               ; preds = %l1, %entry
  %x.0 = phi i32 [ 0, %entry ], [ %inc, %l1 ]
; CHECK: call void @f()
; CHECK-NOT: call void @f()
  call void @f() noduplicate
  %inc = add nsw i32 %x.0, 1
  %exitcond = icmp eq i32 %inc, 3
  br i1 %exitcond, label %l2, label %l1

l2:                                               ; preds = %l1
  ret i32 0
; CHECK: }
}

declare void @f()
