; RUN: opt < %s -correlated-propagation -cvp-dont-add-nowrap-flags=false -S | FileCheck %s

; CHECK-LABEL: @test0(
define void @test0(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 100
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test1(
define void @test1(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, 100
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nuw nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test2(
define void @test2(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, -1
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nuw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test3(
define void @test3(i32 %a) {
entry:
  %cmp = icmp ule i32 %a, -1
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test4(
define void @test4(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 2147483647
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test5(
define void @test5(i32 %a) {
entry:
  %cmp = icmp sle i32 %a, 2147483647
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check for a corner case where an integer value is represented with a constant
; LVILatticeValue instead of constantrange. Check that we don't fail with an
; assertion in this case.
@b = global i32 0, align 4
define void @test6(i32 %a) {
bb:
  %add = add i32 %a, ptrtoint (i32* @b to i32)
  ret void
}

; Check that we can gather information for conditions is the form of
;   and ( i s< 100, Unknown )
; CHECK-LABEL: @test7(
define void @test7(i32 %a, i1 %flag) {
entry:
  %cmp.1 = icmp slt i32 %a, 100
  %cmp = and i1 %cmp.1, %flag
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that we can gather information for conditions is the form of
;   and ( i s< 100, i s> 0 )
; CHECK-LABEL: @test8(
define void @test8(i32 %a) {
entry:
  %cmp.1 = icmp slt i32 %a, 100
  %cmp.2 = icmp sgt i32 %a, 0
  %cmp = and i1 %cmp.1, %cmp.2
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nuw nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that for conditions is the form of cond1 && cond2 we don't mistakenly
; assume that !cond1 && !cond2 holds down to false path.
; CHECK-LABEL: @test8_neg(
define void @test8_neg(i32 %a) {
entry:
  %cmp.1 = icmp sge i32 %a, 100
  %cmp.2 = icmp sle i32 %a, 0
  %cmp = and i1 %cmp.1, %cmp.2
  br i1 %cmp, label %exit, label %bb

bb:
; CHECK: %add = add i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that we can gather information for conditions is the form of
;   and ( i s< 100, and (i s> 0, Unknown )
; CHECK-LABEL: @test9(
define void @test9(i32 %a, i1 %flag) {
entry:
  %cmp.1 = icmp slt i32 %a, 100
  %cmp.2 = icmp sgt i32 %a, 0
  %cmp.3 = and i1 %cmp.2, %flag
  %cmp = and i1 %cmp.1, %cmp.3
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nuw nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that we can gather information for conditions is the form of
;   and ( i s< Unknown, ... )
; CHECK-LABEL: @test10(
define void @test10(i32 %a, i32 %b, i1 %flag) {
entry:
  %cmp.1 = icmp slt i32 %a, %b
  %cmp = and i1 %cmp.1, %flag
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

@limit = external global i32
; CHECK-LABEL: @test11(
define i32 @test11(i32* %p, i32 %i) {
  %limit = load i32, i32* %p, !range !{i32 0, i32 2147483647}
  %within.1 = icmp ugt i32 %limit, %i
  %i.plus.7 = add i32 %i, 7
  %within.2 = icmp ugt i32 %limit, %i.plus.7
  %within = and i1 %within.1, %within.2
  br i1 %within, label %then, label %else

then:
; CHECK: %i.plus.6 = add nuw nsw i32 %i, 6
  %i.plus.6 = add i32 %i, 6
  ret i32 %i.plus.6

else:
  ret i32 0
}

; Check that we can gather information for conditions is the form of
;   or ( i s>= 100, Unknown )
; CHECK-LABEL: @test12(
define void @test12(i32 %a, i1 %flag) {
entry:
  %cmp.1 = icmp sge i32 %a, 100
  %cmp = or i1 %cmp.1, %flag
  br i1 %cmp, label %exit, label %bb

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that we can gather information for conditions is the form of
;   or ( i s>= 100, i s<= 0 )
; CHECK-LABEL: @test13(
define void @test13(i32 %a) {
entry:
  %cmp.1 = icmp sge i32 %a, 100
  %cmp.2 = icmp sle i32 %a, 0
  %cmp = or i1 %cmp.1, %cmp.2
  br i1 %cmp, label %exit, label %bb

bb:
; CHECK: %add = add nuw nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that for conditions is the form of cond1 || cond2 we don't mistakenly
; assume that cond1 || cond2 holds down to true path.
; CHECK-LABEL: @test13_neg(
define void @test13_neg(i32 %a) {
entry:
  %cmp.1 = icmp slt i32 %a, 100
  %cmp.2 = icmp sgt i32 %a, 0
  %cmp = or i1 %cmp.1, %cmp.2
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %add = add i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that we can gather information for conditions is the form of
;   or ( i s>=100, or (i s<= 0, Unknown )
; CHECK-LABEL: @test14(
define void @test14(i32 %a, i1 %flag) {
entry:
  %cmp.1 = icmp sge i32 %a, 100
  %cmp.2 = icmp sle i32 %a, 0
  %cmp.3 = or i1 %cmp.2, %flag
  %cmp = or i1 %cmp.1, %cmp.3
  br i1 %cmp, label %exit, label %bb

bb:
; CHECK: %add = add nuw nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; Check that we can gather information for conditions is the form of
;   or ( i s>= Unknown, ... )
; CHECK-LABEL: @test15(
define void @test15(i32 %a, i32 %b, i1 %flag) {
entry:
  %cmp.1 = icmp sge i32 %a, %b
  %cmp = or i1 %cmp.1, %flag
  br i1 %cmp, label %exit, label %bb

bb:
; CHECK: %add = add nsw i32 %a, 1
  %add = add i32 %a, 1
  br label %exit

exit:
  ret void
}

; single basic block loop
; because the loop exit condition is SLT, we can supplement the iv add
; (iv.next def) with an nsw.
; CHECK-LABEL: @test16(
define i32 @test16(i32* %n, i32* %a) {
preheader:
  br label %loop

loop:
; CHECK: %iv.next = add nsw i32 %iv, 1
  %iv = phi i32 [ 0, %preheader ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %preheader ], [ %acc.curr, %loop ]
  %x = load atomic i32, i32* %a unordered, align 8
  fence acquire
  %acc.curr = add i32 %acc, %x
  %iv.next = add i32 %iv, 1
  %nval = load atomic i32, i32* %n unordered, align 8
  %cmp = icmp slt i32 %iv.next, %nval
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %acc.curr
}
