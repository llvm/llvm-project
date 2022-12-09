; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define void @test0(i32 %init) {
; CHECK-LABEL: Classifying expressions for: @test0
; CHECK: Loop %loop: constant max backedge-taken count is 32
 entry:
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = lshr i32 %iv, 1
  %exit.cond = icmp eq i32 %iv, 0
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test1(i32 %init) {
; CHECK-LABEL: Classifying expressions for: @test1
; CHECK: Loop %loop: constant max backedge-taken count is 32
 entry:
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = shl i32 %iv, 1
  %exit.cond = icmp eq i32 %iv, 0
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test2(i32 %init) {
; CHECK-LABEL: Determining loop execution counts for: @test2
; CHECK: Loop %loop: Unpredictable constant max backedge-taken count.

; Unpredictable because %iv could "stabilize" to either -1 or 0,
; depending on %init.
 entry:
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = ashr i32 %iv, 1
  %exit.cond = icmp eq i32 %iv, 0
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test3(i32* %init.ptr) {
; CHECK-LABEL: Determining loop execution counts for: @test3
; CHECK: Loop %loop: constant max backedge-taken count is 32
 entry:
  %init = load i32, i32* %init.ptr, !range !0
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = ashr i32 %iv, 1
  %exit.cond = icmp eq i32 %iv, 0
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test4(i32* %init.ptr) {
; CHECK-LABEL: Classifying expressions for: @test4
; CHECK-LABEL: Loop %loop: constant max backedge-taken count is 32
 entry:
  %init = load i32, i32* %init.ptr, !range !1
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = ashr i32 %iv, 1
  %exit.cond = icmp eq i32 %iv, -1
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test5(i32* %init.ptr) {
; CHECK-LABEL: Determining loop execution counts for: @test5
; CHECK: Loop %loop: Unpredictable constant max backedge-taken count.

; %iv will "stabilize" to -1, so this is an infinite loop
 entry:
  %init = load i32, i32* %init.ptr, !range !1
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = ashr i32 %iv, 1
  %exit.cond = icmp eq i32 %iv, 0
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test6(i32 %init, i32 %shift.amt) {
; CHECK-LABEL: Determining loop execution counts for: @test6
; CHECK: Loop %loop: Unpredictable constant max backedge-taken count.

; Potentially infinite loop, since %shift.amt could be 0
 entry:
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = lshr i32 %iv, %shift.amt
  %exit.cond = icmp eq i32 %iv, 0
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test7(i32 %init) {
; CHECK-LABEL: Classifying expressions for: @test7
; CHECK: Loop %loop: constant max backedge-taken count is 32

 entry:
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = lshr i32 %iv, 1
  %exit.cond = icmp eq i32 %iv.shift, 0
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test8(i32 %init) {
; CHECK-LABEL: Classifying expressions for: @test8
; CHECK: Loop %loop: Unpredictable constant max backedge-taken count.

; In this test case, %iv.test stabilizes to 127, not -1, so the loop
; is infinite.

 entry:
  br label %loop

 loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.shift, %loop ]
  %iv.shift = ashr i32 %iv, 1
  %iv.test = lshr i32 %iv, 1
  %exit.cond = icmp eq i32 %iv.test, -1
  br i1 %exit.cond, label %leave, label %loop

 leave:
  ret void
}

define void @test9() {
; CHECK-LABEL: Determining loop execution counts for: @test9
; CHECK: Loop %loop: Unpredictable constant max backedge-taken count.

; This is an infinite loop, make sure that it recognized as such.

entry:
  br label %loop

leave:
  ret void

loop:
  %iv = phi i32 [ -20, %entry ], [ %iv.shift, %loop ]
  %iv.shift = ashr i32 %iv, 1
  %exit.cond = icmp sgt i32 %iv, -1
  br i1 %exit.cond, label %leave, label %loop
}

!0 = !{i32 0, i32 50000}
!1 = !{i32 -5000, i32 -1}
