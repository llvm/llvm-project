; RUN: opt -verify-loop-info -irce -S < %s | FileCheck %s
; RUN: opt -verify-loop-info -passes='require<branch-prob>,irce' -S < %s | FileCheck %s

define void @decrementing_loop(i32 *%arr, i32 *%a_len_ptr, i32 %n) {
 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  %start = sub i32 %n, 1
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ %start, %entry ] , [ %idx.dec, %in.bounds ]
  %idx.dec = sub i32 %idx, 1
  %abc.high = icmp slt i32 %idx, %len
  %abc.low = icmp sge i32 %idx, 0
  %abc = and i1 %abc.low, %abc.high
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp sgt i32 %idx.dec, -1
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void

; CHECK: loop.preheader:
; CHECK:   [[len_hiclamp_cmp:[^ ]+]] = icmp slt i32 %len, %n
; CHECK:   [[len_hiclamp:[^ ]+]] = select i1 [[len_hiclamp_cmp]], i32 %len, i32 %n
; CHECK:   [[not_exit_preloop_at_cmp:[^ ]+]] = icmp sgt i32 [[len_hiclamp]], 0
; CHECK:   [[not_exit_preloop_at:[^ ]+]] = select i1 [[not_exit_preloop_at_cmp]], i32 [[len_hiclamp]], i32 0
; CHECK:   %exit.preloop.at = add nsw i32 [[not_exit_preloop_at]], -1
}

; Make sure that we can eliminate the range check when the loop looks like:
; for (i = len.a - 1; i >= 0; --i)
;   b[i] = a[i];
define void @test_01(i32* %a, i32* %b, i32* %a_len_ptr, i32* %b_len_ptr) {

; CHECK-LABEL: test_01
; CHECK:       mainloop:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK:         %rc = and i1 true, true
; CHECK:       loop.preloop:

 entry:
  %len.a = load i32, i32* %a_len_ptr, !range !0
  %len.b = load i32, i32* %b_len_ptr, !range !0
  %first.itr.check = icmp ne i32 %len.a, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ %len.a, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = sub i32 %idx, 1
  %rca = icmp ult i32 %idx.next, %len.a
  %rcb = icmp ult i32 %idx.next, %len.b
  %rc = and i1 %rca, %rcb
  br i1 %rc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %el.a = getelementptr i32, i32* %a, i32 %idx.next
  %el.b = getelementptr i32, i32* %b, i32 %idx.next
  %v = load i32, i32* %el.a
  store i32 %v, i32* %el.b
  %loop.cond = icmp slt i32 %idx, 2
  br i1 %loop.cond, label %exit, label %loop

 out.of.bounds:
  ret void

 exit:
  ret void
}

; Same as test_01, but the latch condition is unsigned
define void @test_02(i32* %a, i32* %b, i32* %a_len_ptr, i32* %b_len_ptr) {

; CHECK-LABEL: test_02
; CHECK:       mainloop:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK:         %rc = and i1 true, true
; CHECK:       loop.preloop:

 entry:
  %len.a = load i32, i32* %a_len_ptr, !range !0
  %len.b = load i32, i32* %b_len_ptr, !range !0
  %first.itr.check = icmp ne i32 %len.a, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ %len.a, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = sub i32 %idx, 1
  %rca = icmp ult i32 %idx.next, %len.a
  %rcb = icmp ult i32 %idx.next, %len.b
  %rc = and i1 %rca, %rcb
  br i1 %rc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %el.a = getelementptr i32, i32* %a, i32 %idx.next
  %el.b = getelementptr i32, i32* %b, i32 %idx.next
  %v = load i32, i32* %el.a
  store i32 %v, i32* %el.b
  %loop.cond = icmp ult i32 %idx, 2
  br i1 %loop.cond, label %exit, label %loop

 out.of.bounds:
  ret void

 exit:
  ret void
}

; Check that we can figure out that IV is non-negative via implication through
; Phi node.
define void @test_03(i32* %a, i32* %a_len_ptr, i1 %cond) {

; CHECK-LABEL: test_03
; CHECK:       mainloop:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK:         br i1 true, label %in.bounds, label %out.of.bounds
; CHECK:       loop.preloop:

 entry:
  %len.a = load i32, i32* %a_len_ptr, !range !0
  %len.minus.one = sub nsw i32 %len.a, 1
  %len.minus.two = sub nsw i32 %len.a, 2
  br i1 %cond, label %if.true, label %if.false

if.true:
  br label %merge

if.false:
  br label %merge

merge:
  %starting.value = phi i32 [ %len.minus.two, %if.true ], [ %len.minus.one, %if.false ]
  %first.itr.check = icmp sgt i32 %len.a, 3
  br i1 %first.itr.check, label %loop, label %exit

loop:
  %idx = phi i32 [ %starting.value, %merge ] , [ %idx.next, %in.bounds ]
  %idx.next = sub i32 %idx, 1
  %rc = icmp ult i32 %idx.next, %len.a
  br i1 %rc, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:
  %el.a = getelementptr i32, i32* %a, i32 %idx.next
  %v = load i32, i32* %el.a
  %loop.cond = icmp slt i32 %idx, 2
  br i1 %loop.cond, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Check that we can figure out that IV is non-negative via implication through
; two Phi nodes.
define void @test_04(i32* %a, i32* %a_len_ptr, i1 %cond) {

; CHECK-LABEL: test_04
; CHECK:       mainloop:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK:         br i1 true, label %in.bounds, label %out.of.bounds
; CHECK:       loop.preloop:

 entry:
  %len.a = load i32, i32* %a_len_ptr, !range !0
  %len.minus.one = sub nsw i32 %len.a, 1
  %len.plus.one = add nsw i32 %len.a, 1
  %len.minus.two = sub nsw i32 %len.a, 2
  br i1 %cond, label %if.true, label %if.false

if.true:
  br label %merge

if.false:
  br label %merge

merge:
  %starting.value = phi i32 [ %len.minus.two, %if.true ], [ %len.minus.one, %if.false ]
  %len.phi = phi i32 [ %len.a, %if.true ], [ %len.plus.one, %if.false ]
  %first.itr.check = icmp sgt i32 %len.a, 3
  br i1 %first.itr.check, label %loop, label %exit

loop:
  %idx = phi i32 [ %starting.value, %merge ] , [ %idx.next, %in.bounds ]
  %idx.next = sub i32 %idx, 1
  %rc = icmp ult i32 %idx.next, %len.phi
  br i1 %rc, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:
  %el.a = getelementptr i32, i32* %a, i32 %idx.next
  %v = load i32, i32* %el.a
  %loop.cond = icmp slt i32 %idx, 2
  br i1 %loop.cond, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Check that we can figure out that IV is non-negative via implication through
; two Phi nodes, one being AddRec.
define void @test_05(i32* %a, i32* %a_len_ptr, i1 %cond) {

; CHECK-LABEL: test_05
; CHECK:       mainloop:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK:         br i1 true, label %in.bounds, label %out.of.bounds
; CHECK:       loop.preloop:

 entry:
  %len.a = load i32, i32* %a_len_ptr, !range !0
  %len.minus.one = sub nsw i32 %len.a, 1
  %len.plus.one = add nsw i32 %len.a, 1
  %len.minus.two = sub nsw i32 %len.a, 2
  br label %merge

merge:
  %starting.value = phi i32 [ %len.minus.two, %entry ], [ %len.minus.one, %merge ]
  %len.phi = phi i32 [ %len.a, %entry ], [ %len.phi.next, %merge ]
  %len.phi.next = add nsw i32 %len.phi, 1
  br i1 true, label %first.iter.check, label %merge

first.iter.check:
  %first.itr.check = icmp sgt i32 %len.a, 3
  br i1 %first.itr.check, label %loop, label %exit

loop:
  %idx = phi i32 [ %starting.value, %first.iter.check ] , [ %idx.next, %in.bounds ]
  %idx.next = sub i32 %idx, 1
  %rc = icmp ult i32 %idx.next, %len.phi
  br i1 %rc, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:
  %el.a = getelementptr i32, i32* %a, i32 %idx.next
  %v = load i32, i32* %el.a
  %loop.cond = icmp slt i32 %idx, 2
  br i1 %loop.cond, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}
