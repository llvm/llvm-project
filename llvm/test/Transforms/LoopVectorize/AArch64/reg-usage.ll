; REQUIRES: asserts

; RUN: opt -mtriple arm64-linux -passes=loop-vectorize -mattr=+sve -debug-only=loop-vectorize,vplan -disable-output <%s 2>&1 | FileCheck %s

; Invariant register usage calculation should take into account if the
; invariant would be used in widened instructions. Only in such cases, a vector
; register would be required for holding the invariant. For all other cases
; such as below(where usage of %0 in loop doesnt require vector register), a
; general purpose register suffices.
; Check that below test doesn't crash while calculating register usage for
; invariant %0

@string = internal unnamed_addr constant [5 x i8] c"abcd\00", align 1

define void @get_invariant_reg_usage(ptr %z) {
; CHECK-LABEL: LV: Checking a loop in 'get_invariant_reg_usage'
; CHECK: LV(REG): VF = 16
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 1 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 3 registers

L.entry:
  %0 = load i128, ptr %z, align 16
  %1 = icmp slt i128 %0, 1
  %a = getelementptr i8, ptr %z, i64 1
  br i1 %1, label %return, label %loopbody

loopbody:                  ;preds = %L.entry, %loopbody
  %b = phi ptr [ %2, %loopbody ], [ @string, %L.entry ]
  %len_input = phi i128 [ %len, %loopbody ], [ %0, %L.entry ]
  %len = add nsw i128 %len_input, -1
  %2 = getelementptr i8, ptr %b, i64 1
  %3 = load i8, ptr %b, align 1
  store i8 %3, ptr %a, align 4
  %.not = icmp eq i128 %len, 0
  br i1 %.not, label %return, label %loopbody

return:                    ;preds = %loopexit, %L.entry
  ret void
}

define void @load_and_compare_only_used_by_assume(ptr %a, ptr noalias %b) {
; CHECK-LABEL: LV: Checking a loop in 'load_and_compare_only_used_by_assume'
; CHECK: LV(REG): VF = vscale x 4
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 3 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 1 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 1 item

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %l.a = load i32, ptr %gep.a
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  %l.b = load i32, ptr %gep.b
  %c = icmp ugt i32 %l.b, 10
  call void @llvm.assume(i1 %c)
  store i32 %l.a, ptr %gep.b
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define dso_local void @dotp_high_register_pressure(ptr %a, ptr %b, ptr %sum, i32 %n) #1 {
; CHECK-LABEL: LV: Checking a loop in 'dotp_high_register_pressure' from <stdin>
; CHECK:       LV(REG): VF = 16
; CHECK-NEXT:  LV(REG): Found max usage: 2 item
; CHECK-NEXT:  LV(REG): RegisterClass: Generic::ScalarRC, 3 registers
; CHECK-NEXT:  LV(REG): RegisterClass: Generic::VectorRC, 48 registers
; CHECK-NEXT:  LV(REG): Found invariant usage: 1 item
entry:
  %cmp100 = icmp sgt i32 %n, 0
  br i1 %cmp100, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %arrayidx13 = getelementptr inbounds nuw i8, ptr %sum, i64 4
  %gep.b.12 = getelementptr inbounds nuw i8, ptr %sum, i64 8
  %arrayidx31 = getelementptr inbounds nuw i8, ptr %sum, i64 12
  %arrayidx40 = getelementptr inbounds nuw i8, ptr %sum, i64 16
  %arrayidx49 = getelementptr inbounds nuw i8, ptr %sum, i64 20
  %arrayidx58 = getelementptr inbounds nuw i8, ptr %sum, i64 24
  %arrayidx67 = getelementptr inbounds nuw i8, ptr %sum, i64 28
  %sum.promoted = load i32, ptr %sum, align 4
  %arrayidx13.promoted = load i32, ptr %arrayidx13, align 4
  %gep.b.12.promoted = load i32, ptr %gep.b.12, align 4
  %arrayidx31.promoted = load i32, ptr %arrayidx31, align 4
  %arrayidx40.promoted = load i32, ptr %arrayidx40, align 4
  %arrayidx49.promoted = load i32, ptr %arrayidx49, align 4
  %arrayidx58.promoted = load i32, ptr %arrayidx58, align 4
  %arrayidx67.promoted = load i32, ptr %arrayidx67, align 4
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:              ; preds = %for.body
  %add.lcssa = phi i32 [ %add.1, %for.body ]
  %add.2.lcssa = phi i32 [ %add.2, %for.body ]
  %add.3.lcssa = phi i32 [ %add.3, %for.body ]
  %add.4.lcssa = phi i32 [ %add.4, %for.body ]
  %add.5.lcssa = phi i32 [ %add.5, %for.body ]
  %add.6.lcssa = phi i32 [ %add.6, %for.body ]
  %add.7.lcssa = phi i32 [ %add.7, %for.body ]
  %add.8.lcssa = phi i32 [ %add.8, %for.body ]
  store i32 %add.lcssa, ptr %sum, align 4
  store i32 %add.2.lcssa, ptr %arrayidx13, align 4
  store i32 %add.3.lcssa, ptr %gep.b.12, align 4
  store i32 %add.4.lcssa, ptr %arrayidx31, align 4
  store i32 %add.5.lcssa, ptr %arrayidx40, align 4
  store i32 %add.6.lcssa, ptr %arrayidx49, align 4
  store i32 %add.7.lcssa, ptr %arrayidx58, align 4
  store i32 %add.8.lcssa, ptr %arrayidx67, align 4
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.for.cond.cleanup_crit_edge, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %0 = phi i32 [ %arrayidx67.promoted, %for.body.lr.ph ], [ %add.8, %for.body ]
  %1 = phi i32 [ %arrayidx58.promoted, %for.body.lr.ph ], [ %add.7, %for.body ]
  %2 = phi i32 [ %arrayidx49.promoted, %for.body.lr.ph ], [ %add.6, %for.body ]
  %3 = phi i32 [ %arrayidx40.promoted, %for.body.lr.ph ], [ %add.5, %for.body ]
  %4 = phi i32 [ %arrayidx31.promoted, %for.body.lr.ph ], [ %add.4, %for.body ]
  %5 = phi i32 [ %gep.b.12.promoted, %for.body.lr.ph ], [ %add.3, %for.body ]
  %6 = phi i32 [ %arrayidx13.promoted, %for.body.lr.ph ], [ %add.2, %for.body ]
  %7 = phi i32 [ %sum.promoted, %for.body.lr.ph ], [ %add.1, %for.body ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i64 %indvars.iv
  %load.a = load i8, ptr %arrayidx, align 1
  %ext.a = zext i8 %load.a to i32
  %9 = shl nsw i64 %indvars.iv, 3
  %gep.b.1 = getelementptr inbounds nuw i8, ptr %b, i64 %9
  %load.b.1 = load i8, ptr %gep.b.1, align 1
  %ext.b.1 = sext i8 %load.b.1 to i32
  %mul.1 = mul nsw i32 %ext.b.1, %ext.a
  %add.1 = add nsw i32 %mul.1, %7
  %11 = or disjoint i64 %9, 1
  %gep.b.2 = getelementptr inbounds nuw i8, ptr %b, i64 %11
  %load.b.2 = load i8, ptr %gep.b.2, align 1
  %ext.b.2 = sext i8 %load.b.2 to i32
  %mul.2 = mul nsw i32 %ext.b.2, %ext.a
  %add.2 = add nsw i32 %mul.2, %6
  %13 = or disjoint i64 %9, 2
  %gep.b.3 = getelementptr inbounds nuw i8, ptr %b, i64 %13
  %load.b.3 = load i8, ptr %gep.b.3, align 1
  %ext.b.3 = sext i8 %load.b.3 to i32
  %mul.3 = mul nsw i32 %ext.b.3, %ext.a
  %add.3 = add nsw i32 %mul.3, %5
  %15 = or disjoint i64 %9, 3
  %gep.b.4 = getelementptr inbounds nuw i8, ptr %b, i64 %15
  %load.b.4 = load i8, ptr %gep.b.4, align 1
  %ext.b.4 = sext i8 %load.b.4 to i32
  %mul.4 = mul nsw i32 %ext.b.4, %ext.a
  %add.4 = add nsw i32 %mul.4, %4
  %17 = or disjoint i64 %9, 4
  %gep.b.5 = getelementptr inbounds nuw i8, ptr %b, i64 %17
  %load.b.5 = load i8, ptr %gep.b.5, align 1
  %ext.b.5 = sext i8 %load.b.5 to i32
  %mul.5 = mul nsw i32 %ext.b.5, %ext.a
  %add.5 = add nsw i32 %mul.5, %3
  %19 = or disjoint i64 %9, 5
  %gep.b.6 = getelementptr inbounds nuw i8, ptr %b, i64 %19
  %load.b.6 = load i8, ptr %gep.b.6, align 1
  %ext.b.6 = sext i8 %load.b.6 to i32
  %mul.6 = mul nsw i32 %ext.b.6, %ext.a
  %add.6 = add nsw i32 %mul.6, %2
  %21 = or disjoint i64 %9, 6
  %gep.b.7 = getelementptr inbounds nuw i8, ptr %b, i64 %21
  %load.b.7 = load i8, ptr %gep.b.7, align 1
  %ext.b.7 = sext i8 %load.b.7 to i32
  %mul.7 = mul nsw i32 %ext.b.7, %ext.a
  %add.7 = add nsw i32 %mul.7, %1
  %23 = or disjoint i64 %9, 7
  %gep.b.8 = getelementptr inbounds nuw i8, ptr %b, i64 %23
  %load.b.8 = load i8, ptr %gep.b.8, align 1
  %ext.b.8 = sext i8 %load.b.8 to i32
  %mul.8 = mul nsw i32 %ext.b.8, %ext.a
  %add.8 = add nsw i32 %mul.8, %0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}

define i32 @dotp_unrolled(i32 %num_out, i64 %num_in, ptr %a, ptr %b) {
; CHECK-LABEL: LV: Checking a loop in 'dotp_unrolled' from <stdin>
; CHECK:       LV(REG): VF = 16
; CHECK-NEXT:  LV(REG): Found max usage: 2 item
; CHECK-NEXT:  LV(REG): RegisterClass: Generic::ScalarRC, 9 registers
; CHECK-NEXT:  LV(REG): RegisterClass: Generic::VectorRC, 24 registers
; CHECK-NEXT:  LV(REG): Found invariant usage: 1 item
entry:
  br label %for.body

for.body:                                    ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %accum3 = phi i32 [ 0, %entry ], [ %add.a3, %for.body ]
  %accum2 = phi i32 [ 0, %entry ], [ %add.a2, %for.body ]
  %accum1 = phi i32 [ 0, %entry ], [ %add.a1, %for.body ]
  %accum0 = phi i32 [ 0, %entry ], [ %add.a0, %for.body ]
  %gep.a0 = getelementptr inbounds i8, ptr %a, i64 %iv
  %gep.b0 = getelementptr inbounds i8, ptr %b, i64 %iv
  %offset.1 = or disjoint i64 %iv, 1
  %gep.a1 = getelementptr inbounds i8, ptr %a, i64 %offset.1
  %gep.b1 = getelementptr inbounds i8, ptr %b, i64 %offset.1
  %offset.2 = or disjoint i64 %iv, 2
  %gep.a2 = getelementptr inbounds i8, ptr %a, i64 %offset.2
  %gep.b2 = getelementptr inbounds i8, ptr %b, i64 %offset.2
  %offset.3 = or disjoint i64 %iv, 3
  %gep.a3 = getelementptr inbounds i8, ptr %a, i64 %offset.3
  %gep.b3 = getelementptr inbounds i8, ptr %b, i64 %offset.3
  %load.a0 = load i8, ptr %gep.a0, align 1
  %ext.a0 = sext i8 %load.a0 to i32
  %load.b0 = load i8, ptr %gep.b0, align 1
  %ext.b0 = sext i8 %load.b0 to i32
  %mul.a0 = mul nsw i32 %ext.b0, %ext.a0
  %add.a0 = add nsw i32 %mul.a0, %accum0
  %load.a1 = load i8, ptr %gep.a1, align 1
  %ext.a1 = sext i8 %load.a1 to i32
  %load.b1 = load i8, ptr %gep.b1, align 1
  %ext.b1 = sext i8 %load.b1 to i32
  %mul.a1 = mul nsw i32 %ext.a1, %ext.b1
  %add.a1 = add nsw i32 %mul.a1, %accum1
  %load.a2 = load i8, ptr %gep.a2, align 1
  %ext.a2 = sext i8 %load.a2 to i32
  %load.b2 = load i8, ptr %gep.b2, align 1
  %ext.b2 = sext i8 %load.b2 to i32
  %mul.a2 = mul nsw i32 %ext.a2, %ext.b2
  %add.a2 = add nsw i32 %mul.a2, %accum2
  %load.a3 = load i8, ptr %gep.a3, align 1
  %ext.a3 = sext i8 %load.a3 to i32
  %load.b3 = load i8, ptr %gep.b3, align 1
  %ext.b3 = sext i8 %load.b3 to i32
  %mul.a3 = mul nsw i32 %ext.a3, %ext.b3
  %add.a3 = add nsw i32 %mul.a3, %accum3
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %num_in
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                        ; preds = %for.body
  %result0 = add nsw i32 %add.a0, %add.a1
  %result1 = add nsw i32 %add.a2, %add.a3
  %result = add nsw i32 %result0, %result1
  ret i32 %result
}
