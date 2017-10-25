; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s

; The test demonstrates that incorrect behavior of Clamp may lead to incorrect
; calculation of post-loop exit condition.

; CHECK: irce: in function test: constrained Loop at depth 1 containing: %loop<header><exiting>,%in_bounds<exiting>,%not_zero<latch><exiting>

define void @test() {
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
