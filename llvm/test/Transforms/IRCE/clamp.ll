; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s

; The test demonstrates that incorrect behavior of Clamp may lead to incorrect
; calculation of post-loop exit condition.

; CHECK-LABEL: irce: in function test_01: constrained Loop at depth 1 containing: %loop<header><exiting>,%in_bounds<exiting>,%not_zero<latch><exiting>
; CHECK-LABEL: irce: in function test_02: constrained Loop at depth 1 containing: %loop<header><exiting>,%in_bounds<latch><exiting>

define void @test_01() {

; CHECK-LABEL: test_01

entry:
  %indvars.iv.next467 = add nuw nsw i64 2, 1
  %length.i167 = load i32, i32 addrspace(1)* undef, align 8
  %tmp21 = zext i32 %length.i167 to i64
  %tmp34 = load atomic i32, i32 addrspace(1)* undef unordered, align 4
  %tmp35 = add i32 %tmp34, -9581
  %tmp36 = icmp ugt i32 %length.i167, 1
  br i1 %tmp36, label %preheader, label %exit

exit:                                          ; preds = %in_bounds, %loop, %not_zero, %entry
  ret void

preheader:                                 ; preds = %entry
; CHECK:      preheader:
; CHECK-NEXT:   %length_gep.i146 = getelementptr inbounds i8, i8 addrspace(1)* undef, i64 8
; CHECK-NEXT:   %length_gep_typed.i147 = bitcast i8 addrspace(1)* undef to i32 addrspace(1)*
; CHECK-NEXT:   %tmp43 = icmp ult i64 %indvars.iv.next467, %tmp21
; CHECK-NEXT:   [[C0:%[^ ]+]] = icmp ugt i64 %tmp21, 1
; CHECK-NEXT:   %exit.mainloop.at = select i1 [[C0]], i64 %tmp21, i64 1
; CHECK-NEXT:   [[C1:%[^ ]+]] = icmp ult i64 1, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[C1]], label %loop.preheader, label %main.pseudo.exit

  %length_gep.i146 = getelementptr inbounds i8, i8 addrspace(1)* undef, i64 8
  %length_gep_typed.i147 = bitcast i8 addrspace(1)* undef to i32 addrspace(1)*
  %tmp43 = icmp ult i64 %indvars.iv.next467, %tmp21
  br label %loop

not_zero:                                       ; preds = %in_bounds
; CHECK:      not_zero:
; CHECK:        %tmp56 = icmp ult i64 %indvars.iv.next, %tmp21
; CHECK-NEXT:   [[COND:%[^ ]+]] = icmp ult i64 %indvars.iv.next, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[COND]], label %loop, label %main.exit.selector

  %tmp51 = trunc i64 %indvars.iv.next to i32
  %tmp53 = mul i32 %tmp51, %tmp51
  %tmp54 = add i32 %tmp53, -9582
  %tmp55 = add i32 %tmp54, %tmp62
  %tmp56 = icmp ult i64 %indvars.iv.next, %tmp21
  br i1 %tmp56, label %loop, label %exit

loop:                                       ; preds = %not_zero, %preheader
  %tmp62 = phi i32 [ 1, %preheader ], [ %tmp55, %not_zero ]
  %indvars.iv750 = phi i64 [ 1, %preheader ], [ %indvars.iv.next, %not_zero ]
  %length.i148 = load i32, i32 addrspace(1)* %length_gep_typed.i147, align 8
  %tmp68 = zext i32 %length.i148 to i64
  %tmp97 = icmp ult i64 2, %tmp68
  %or.cond = and i1 %tmp43, %tmp97
  %tmp99 = icmp ult i64 %indvars.iv750, %tmp21
  %or.cond1 = and i1 %or.cond, %tmp99
  br i1 %or.cond1, label %in_bounds, label %exit

in_bounds:                                       ; preds = %loop
  %indvars.iv.next = add nuw nsw i64 %indvars.iv750, 3
  %tmp107 = icmp ult i64 %indvars.iv.next, 2
  br i1 %tmp107, label %not_zero, label %exit
}

define void @test_02() {

; CHECK-LABEL: test_02
; CHECK-NOT:   br i1 false, label %loop.preloop.preheader, label %preloop.pseudo.exit
; CHECK-NOT:   postloop
; CHECK:       entry:
; CHECK-NEXT:    br i1 true, label %loop.preloop.preheader, label %preloop.pseudo.exit
; CHECK:       mainloop:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %iv1 = phi i64 [ %iv1.preloop.copy, %mainloop ], [ %iv1.next, %in_bounds ]
; CHECK-NEXT:    %iv2 = phi i64 [ %iv2.preloop.copy, %mainloop ], [ %iv2.next, %in_bounds ]
; CHECK-NEXT:    %iv2.offset = add i64 %iv2, 1
; CHECK-NEXT:    %rc = icmp ult i64 %iv2.offset, 400
; CHECK-NEXT:    br i1 true, label %in_bounds, label %bci_321.loopexit1
; CHECK:       in_bounds:
; CHECK-NEXT:    %iv1.next = add nuw nsw i64 %iv1, 2
; CHECK-NEXT:    %iv2.next = add nuw nsw i64 %iv2, 2
; CHECK-NEXT:    %cond = icmp ugt i64 %iv1, 204
; CHECK-NEXT:    br i1 %cond, label %bci_321.loopexit1, label %loop
; CHECK:       loop.preloop:
; CHECK-NEXT:    %iv1.preloop = phi i64 [ %iv1.next.preloop, %in_bounds.preloop ], [ 3, %loop.preloop.preheader ]
; CHECK-NEXT:    %iv2.preloop = phi i64 [ %iv2.next.preloop, %in_bounds.preloop ], [ 4294967295, %loop.preloop.preheader ]
; CHECK-NEXT:    %iv2.offset.preloop = add i64 %iv2.preloop, 1
; CHECK-NEXT:    %rc.preloop = icmp ult i64 %iv2.offset.preloop, 400
; CHECK-NEXT:    br i1 %rc.preloop, label %in_bounds.preloop, label %bci_321.loopexit
; CHECK:       in_bounds.preloop:
; CHECK-NEXT:    %iv1.next.preloop = add nuw nsw i64 %iv1.preloop, 2
; CHECK-NEXT:    %iv2.next.preloop = add nuw nsw i64 %iv2.preloop, 2
; CHECK-NEXT:    %cond.preloop = icmp ugt i64 %iv1.preloop, 204
; CHECK-NEXT:    [[C0:%[^ ]+]] = icmp ult i64 %iv1.preloop, 205
; CHECK-NEXT:    [[C1:%[^ ]+]] = xor i1 [[C0]], true
; CHECK-NEXT:    br i1 [[C1]], label %preloop.exit.selector, label %loop.preloop
; CHECK:       preloop.pseudo.exit:
; CHECK-NEXT:    %iv1.preloop.copy = phi i64 [ 3, %entry ], [ %iv1.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:    %iv2.preloop.copy = phi i64 [ 4294967295, %entry ], [ %iv2.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:    %indvar.end = phi i64 [ 1, %entry ], [ %iv1.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:    br label %mainloop

entry:
  br label %loop

loop:                                    ; preds = %in_bounds, %entry
  %iv1 = phi i64 [ 3, %entry ], [ %iv1.next, %in_bounds ]
  %iv2 = phi i64 [ 4294967295, %entry ], [ %iv2.next, %in_bounds ]
  %iv2.offset = add i64 %iv2, 1
  %rc = icmp ult i64 %iv2.offset, 400
  br i1 %rc, label %in_bounds, label %bci_321

bci_321:                                          ; preds = %in_bounds, %loop
  ret void

in_bounds:                                 ; preds = %loop
  %iv1.next = add nuw nsw i64 %iv1, 2
  %iv2.next = add nuw nsw i64 %iv2, 2
  %cond = icmp ugt i64 %iv1, 204
  br i1 %cond, label %bci_321, label %loop
}
