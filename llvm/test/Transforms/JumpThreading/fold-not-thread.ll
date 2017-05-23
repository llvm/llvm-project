; RUN: opt -jump-threading -S -verify < %s | FileCheck %s

declare i32 @f1()
declare i32 @f2()
declare void @f3()
declare void @f4(i32)


; Make sure we update the phi node properly.
;
; CHECK-LABEL: define void @test_br_folding_not_threading_update_phi(
; CHECK: br label %L1
; Make sure we update the phi node properly here, i.e. we only have 2 predecessors, entry and L0
; CHECK: %res.0 = phi i32 [ 0, %L0 ], [ 1, %entry ]
define void @test_br_folding_not_threading_update_phi(i32 %val) nounwind {
entry:
  %cmp = icmp eq i32 %val, 32
  br i1 %cmp, label %L0, label %L1
L0:
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  switch i32 %val, label %L2 [
    i32 0, label %L1
    i32 32, label %L1
  ]

L1:
	%res.0 = phi i32 [ 0, %L0 ], [ 0, %L0 ], [1, %entry]
  call void @f4(i32 %res.0)
  ret void
L2:
  call void @f3()
  ret void
}

; Make sure we can fold this branch ... We will not be able to thread it as
; L0 is too big to duplicate. L2 is the unreachable block here.
;
; CHECK-LABEL: @test_br_folding_not_threading(
; CHECK: L1:
; CHECK: call i32 @f2()
; CHECK: call void @f3()
; CHECK-NEXT: ret void
; CHECK-NOT: br
; CHECK: L3:
define void @test_br_folding_not_threading(i1 %cond) nounwind {
entry:
  br i1 %cond, label %L0, label %L3 
L0:
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  br i1 %cond, label %L1, label %L2 

L1:
  call void @f3()
  ret void
L2:
  call void @f3()
  ret void
L3:
  call void @f3()
  ret void
}


; Make sure we can fold this branch ... We will not be able to thread it as
; L0 is too big to duplicate. L2 is the unreachable block here.
; With more than 1 predecessors.
;
; CHECK-LABEL: @test_br_folding_not_threading_multiple_preds(
; CHECK: L1:
; CHECK: call i32 @f2()
; CHECK: call void @f3()
; CHECK-NEXT: ret void
; CHECK-NOT: br
; CHECK: L3:
define void @test_br_folding_not_threading_multiple_preds(i1 %condx, i1 %cond) nounwind {
entry:
  br i1 %condx, label %X0, label %X1

X0:
  br i1 %cond, label %L0, label %L3 

X1:
  br i1 %cond, label %L0, label %L3 

L0:
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  br i1 %cond, label %L1, label %L2 

L1:
  call void @f3()
  ret void
L2:
  call void @f3()
  ret void
L3:
  call void @f3()
  ret void
}

; Make sure we can do the RAUW for %add...
;
; CHECK-LABEL: @rauw_if_possible(
; CHECK: call void @f4(i32 96)
define void @rauw_if_possible(i32 %value) nounwind {
entry:
  %cmp = icmp eq i32 %value, 32
  br i1 %cmp, label %L0, label %L3 
L0:
  call i32 @f2()
  call i32 @f2()
  %add = add i32 %value, 64
  switch i32 %add, label %L3 [
    i32 32, label %L1
    i32 96, label %L2
    ]

L1:
  call void @f3()
  ret void
L2:
  call void @f4(i32 %add)
  ret void
L3:
  call void @f3()
  ret void
}

; Make sure we can NOT do the RAUW for %add...
;
; CHECK-LABEL: @rauw_if_possible2(
; CHECK: call void @f4(i32 %add) 
define void @rauw_if_possible2(i32 %value) nounwind {
entry:
  %cmp = icmp eq i32 %value, 32
  %add = add i32 %value, 64
  br i1 %cmp, label %L0, label %L2 
L0:
  call i32 @f2()
  call i32 @f2()
  switch i32 %add, label %L3 [
    i32 32, label %L1
    i32 96, label %L2
    ]

L1:
  call void @f3()
  ret void
L2:
  call void @f4(i32 %add)
  ret void
L3:
  call void @f3()
  ret void
}

; Make sure we can fold this branch ... We will not be able to thread it as
; L0 is too big to duplicate.
; We do not attempt to rewrite the indirectbr target here, but we still take
; its target after L0 into account and that enables us to fold.
;
; L2 is the unreachable block here.
; 
; CHECK-LABEL: @test_br_folding_not_threading_indirect_branch(
; CHECK: L1:
; CHECK: call i32 @f2()
; CHECK: call void @f3()
; CHECK-NEXT: ret void
; CHECK-NOT: br
; CHECK: L3:
define void @test_br_folding_not_threading_indirect_branch(i1 %condx, i1 %cond) nounwind {
entry:
  br i1 %condx, label %X0, label %X1

X0:
  br i1 %cond, label %L0, label %L3

X1:
  br i1 %cond, label %XX1, label %L3

XX1:
  indirectbr i8* blockaddress(@test_br_folding_not_threading_indirect_branch, %L0), [label %L0]

L0:
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  call i32 @f2()
  br i1 %cond, label %L1, label %L2

L1:
  call void @f3()
  ret void

L2:
  call void @f3()
  ret void

L3:
  call void @f3()
  ret void
}
