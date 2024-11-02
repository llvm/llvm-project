; RUN: opt -verify-loop-info -irce -S < %s | FileCheck %s
; RUN: opt -verify-loop-info -passes='require<branch-prob>,irce' -S < %s | FileCheck %s

define void @single_access_no_preloop_no_offset(ptr %arr, ptr %a_len_ptr, i32 %n) {
 entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; CHECK-LABEL: @single_access_no_preloop_no_offset(

; CHECK: loop:
; CHECK: br i1 true, label %in.bounds, label %out.of.bounds

; CHECK: main.exit.selector:
; CHECK-NEXT: %idx.next.lcssa = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT: [[continue:%[^ ]+]] = icmp slt i32 %idx.next.lcssa, %n
; CHECK-NEXT: br i1 [[continue]], label %main.pseudo.exit, label %exit.loopexit

; CHECK: main.pseudo.exit:
; CHECK-NEXT: %idx.copy = phi i32 [ 0, %loop.preheader ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT: %indvar.end = phi i32 [ 0, %loop.preheader ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT: br label %postloop

; CHECK: postloop:
; CHECK-NEXT: br label %loop.postloop

; CHECK: loop.postloop:
; CHECK-NEXT: %idx.postloop = phi i32 [ %idx.next.postloop, %in.bounds.postloop ], [ %idx.copy, %postloop ]
; CHECK-NEXT: %idx.next.postloop = add i32 %idx.postloop, 1
; CHECK-NEXT: %abc.postloop = icmp slt i32 %idx.postloop, %len
; CHECK-NEXT: br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds

; CHECK: in.bounds.postloop:
; CHECK-NEXT: %addr.postloop = getelementptr i32, ptr %arr, i32 %idx.postloop
; CHECK-NEXT: store i32 0, ptr %addr.postloop
; CHECK-NEXT: %next.postloop = icmp slt i32 %idx.next.postloop, %n
; CHECK-NEXT: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit


define void @single_access_no_preloop_with_offset(ptr %arr, ptr %a_len_ptr, i32 %n) {
 entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.for.abc = add i32 %idx, 4
  %abc = icmp slt i32 %idx.for.abc, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx.for.abc
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; CHECK-LABEL: @single_access_no_preloop_with_offset(

; CHECK: loop.preheader:
; CHECK: [[safe_range_end:[^ ]+]] = add nsw i32 %len, -4
; CHECK: [[exit_main_loop_at_hiclamp:[^ ]+]] = call i32 @llvm.smin.i32(i32 %n, i32 [[safe_range_end]])
; CHECK: [[exit_main_loop_at_loclamp:[^ ]+]] = call i32 @llvm.smax.i32(i32 [[exit_main_loop_at_hiclamp]], i32 0)
; CHECK: [[enter_main_loop:[^ ]+]] = icmp slt i32 0, [[exit_main_loop_at_loclamp]]
; CHECK: br i1 [[enter_main_loop]], label %[[loop_preheader:[^ ,]+]], label %main.pseudo.exit

; CHECK: loop:
; CHECK: br i1 true, label %in.bounds, label %out.of.bounds

; CHECK: in.bounds:
; CHECK: [[continue_main_loop:[^ ]+]] = icmp slt i32 %idx.next, [[exit_main_loop_at_loclamp]]
; CHECK: br i1 [[continue_main_loop]], label %loop, label %main.exit.selector

; CHECK: main.pseudo.exit:
; CHECK:  %idx.copy = phi i32 [ 0, %loop.preheader ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK:  br label %postloop

; CHECK: loop.postloop:
; CHECK: %idx.postloop = phi i32 [ %idx.next.postloop, %in.bounds.postloop ], [ %idx.copy, %postloop ]

; CHECK: in.bounds.postloop:
; CHECK: %next.postloop = icmp slt i32 %idx.next.postloop, %n
; CHECK: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

; Make sure that we do not do IRCE if we know that the safe iteration range of
; the main loop is empty.

define void @single_access_empty_range(ptr %arr, ptr %a_len_ptr, i32 %n) {
 entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, 0
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; CHECK-LABEL: @single_access_empty_range(
; CHECK-NOT:   br i1 false
; CHECK-NOT:   preloop
; CHECK-NOT:   postloop

define void @single_access_empty_range_2(ptr %arr, ptr %a_len_ptr, i32 %n) {
 entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds2 ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, 60
  br i1 %abc, label %in.bounds1, label %out.of.bounds, !prof !1

 in.bounds1:
  %def = icmp slt i32 %idx, 0
  br i1 %def, label %in.bounds2, label %out.of.bounds, !prof !1

in.bounds2:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; CHECK-LABEL: @single_access_empty_range_2(
; CHECK-NOT:   br i1 false
; CHECK-NOT:   preloop

define void @single_access_no_preloop_no_offset_phi_len(ptr %arr, ptr %a_len_ptr, ptr %b_len_ptr, i32 %n, i1 %unknown_cond) {
 entry:
  br i1 %unknown_cond, label %if.true, label %if.false

if.true:
  %len_a = load i32, ptr %a_len_ptr, !range !0
  br label %merge

if.false:
  %len_b = load i32, ptr %b_len_ptr, !range !0
  br label %merge

merge:
  %len = phi i32 [ %len_a, %if.true ], [ %len_b, %if.false ]
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %merge ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; CHECK-LABEL: @single_access_no_preloop_no_offset_phi_len(

; CHECK: loop:
; CHECK: br i1 true, label %in.bounds, label %out.of.bounds

; CHECK: main.exit.selector:
; CHECK-NEXT: %idx.next.lcssa = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT: [[continue:%[^ ]+]] = icmp slt i32 %idx.next.lcssa, %n
; CHECK-NEXT: br i1 [[continue]], label %main.pseudo.exit, label %exit.loopexit

; CHECK: main.pseudo.exit:
; CHECK-NEXT: %idx.copy = phi i32 [ 0, %loop.preheader ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT: %indvar.end = phi i32 [ 0, %loop.preheader ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT: br label %postloop

; CHECK: postloop:
; CHECK-NEXT: br label %loop.postloop

; CHECK: loop.postloop:
; CHECK-NEXT: %idx.postloop = phi i32 [ %idx.next.postloop, %in.bounds.postloop ], [ %idx.copy, %postloop ]
; CHECK-NEXT: %idx.next.postloop = add i32 %idx.postloop, 1
; CHECK-NEXT: %abc.postloop = icmp slt i32 %idx.postloop, %len
; CHECK-NEXT: br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds

; CHECK: in.bounds.postloop:
; CHECK-NEXT: %addr.postloop = getelementptr i32, ptr %arr, i32 %idx.postloop
; CHECK-NEXT: store i32 0, ptr %addr.postloop
; CHECK-NEXT: %next.postloop = icmp slt i32 %idx.next.postloop, %n
; CHECK-NEXT: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}
