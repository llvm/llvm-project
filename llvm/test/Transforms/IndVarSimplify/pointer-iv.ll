; RUN: opt < %s -passes=indvars -indvars-eliminate-pointer-ivs=true -S | FileCheck %s

; Comprehensive test coverage for pointer IV elimination in IndVarSimplify.

@global_array = global [1024 x i8] zeroinitializer
%struct.Point = type { i32, i32 }
declare void @use_value(i64)

; Positive Cases

; TEST: Basic pointer IV transformation
define void @test_basic_transformation(ptr %base, i32 %n) {
; CHECK-LABEL: @test_basic_transformation(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64 [ 0, {{.*}} ], [ [[IV_NEXT:%.*]], %loop ]
; CHECK:         [[IV_NEXT]] = add nuw nsw i64 [[IV]], 1
; CHECK:         [[PTR:%.*]] = getelementptr i8, ptr %base, i64 [[IV]]
; CHECK:         store i8 42, ptr [[PTR]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Stride != 1
define void @test_stride_4(ptr %base, i32 %n) {
; CHECK-LABEL: @test_stride_4(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[SCALED:%.*]] = mul {{.*}} i64 [[IV]], 4
; CHECK:         [[PTR:%.*]] = getelementptr i32, ptr %base, i64 [[SCALED]]
; CHECK:         store i32 0, ptr [[PTR]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i32 0, ptr %p, align 4
  %p.next = getelementptr inbounds i32, ptr %p, i64 4
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Negative stride
define void @test_negative_stride(ptr %end, i32 %n) {
; CHECK-LABEL: @test_negative_stride(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[SCALED:%.*]] = mul {{.*}} i64 [[IV]], -1
; CHECK:         [[PTR:%.*]] = getelementptr i8, ptr %end, i64 [[SCALED]]
; CHECK:         store i8 0, ptr [[PTR]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %end, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 0, ptr %p, align 1
  %p.next = getelementptr inbounds i8, ptr %p, i64 -1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: GEP without nuw flag
; Note: Other passes may still add nuw/nsw after our transformation,
; so we just verify the transformation happens correctly.
define void @test_no_nuw_flag(ptr %base, i32 %n) {
; CHECK-LABEL: @test_no_nuw_flag(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[IV_NEXT:%.*]] = add {{.*}} i64 [[IV]], 1
; CHECK:         [[PTR:%.*]] = getelementptr i8, ptr %base, i64 [[IV]]
; CHECK:         store i8 42, ptr [[PTR]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  ; GEP without nuw flag
  %p.next = getelementptr inbounds i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Base pointer is not an instruction (function argument)
define void @test_base_is_argument(ptr %base, i32 %n) {
; Function argument as base - SHOULD be transformed
; CHECK-LABEL: @test_base_is_argument(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK:         [[PTR:%.*]] = getelementptr {{.*}} ptr %base, i64 [[IV]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Base pointer is global constant (not instruction)
define void @test_base_is_global(i32 %n) {
; Global as base - SHOULD be transformed (not an instruction)
; CHECK-LABEL: @test_base_is_global(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64 [ 0, {{.*}} ]
; CHECK:         [[PTR:%.*]] = getelementptr {{.*}} ptr @global_array, i64 [[IV]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ @global_array, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Base pointer from same loop (should transform)
define void @test_base_from_same_loop_header(ptr %input, i32 %n) {
; Base computed in same loop header - SHOULD be transformed
; CHECK-LABEL: @test_base_from_same_loop_header(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[PTR:%.*]] = getelementptr {{.*}} ptr %input, i64 [[IV]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %input, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Multiple uses of PHI in loop body
define void @test_multiple_phi_uses(ptr %base, i32 %n) {
; Multiple uses of the pointer PHI in loop body
; CHECK-LABEL: @test_multiple_phi_uses(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[PTR1:%.*]] = getelementptr {{.*}} ptr %base, i64 [[IV]]
; CHECK:         load i8, ptr [[PTR1]]
; CHECK:         [[PTR2:%.*]] = getelementptr {{.*}} ptr %base, i64 [[IV]]
; CHECK:         store i8 {{.*}}, ptr [[PTR2]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  ; First use of %p
  %val = load i8, ptr %p, align 1
  %inc = add i8 %val, 1
  ; Second use of %p
  store i8 %inc, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Increment used multiple times in loop body
define void @test_multiple_incr_uses(ptr %base, i32 %n) {
; Multiple uses of the increment GEP in loop body
; CHECK-LABEL: @test_multiple_incr_uses(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[IV_NEXT:%.*]] = add {{.*}} i64 [[IV]], 1
; CHECK:         [[NEXT_PTR1:%.*]] = getelementptr {{.*}} ptr %base, i64 [[IV_NEXT]]
; CHECK:         load i8, ptr [[NEXT_PTR1]]
; CHECK:         [[NEXT_PTR2:%.*]] = getelementptr {{.*}} ptr %base, i64 [[IV_NEXT]]
; CHECK:         store i8 {{.*}}, ptr [[NEXT_PTR2]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  ; First use of %p.next (lookahead)
  %next.val = load i8, ptr %p.next, align 1
  %inc = add i8 %next.val, 1
  ; Second use of %p.next
  store i8 %inc, ptr %p.next, align 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Non-unit stride (should now be optimized with multiplication)
define void @test_non_unit_stride(ptr %A, ptr %last, ptr %B) {
; CHECK-LABEL: @test_non_unit_stride(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[B_PHI_INT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], 2
; CHECK-NEXT:    [[B_COMPUTED:%.*]] = getelementptr i32, ptr [[B:%.*]], i64 [[B_PHI_INT_SCALED]]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[B_COMPUTED]], align 4
; CHECK-NEXT:    [[A_PHI_INT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], 2
; CHECK-NEXT:    [[A_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_SCALED]]
; CHECK-NEXT:    store i32 [[VAL]], ptr [[A_COMPUTED]], align 4
; CHECK-NEXT:    [[A_PHI_INT_NEXT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT_NEXT]], 2
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_NEXT_SCALED]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %val = load i32, ptr %B.phi, align 4
  store i32 %val, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 2  ; stride of 2, not 1
  %B.next = getelementptr inbounds i32, ptr %B.phi, i64 2
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Multiple uses of GEP
define void @test_multiple_gep_uses(ptr %A, ptr %last, ptr %B) {
; CHECK-LABEL: @test_multiple_gep_uses(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP1:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP1]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[B_PHI:%.*]] = getelementptr i32, ptr [[B:%.*]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[B_PHI]], align 4
; CHECK-NEXT:    [[A_PHI:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    store i32 [[VAL]], ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[A_NEXT:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_NEXT]]
; CHECK-NEXT:    [[EXTRA_USE:%.*]] = ptrtoint ptr [[A_NEXT]] to i64
; CHECK-NEXT:    call void @use_value(i64 [[EXTRA_USE]])
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_NEXT]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP1]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %val = load i32, ptr %B.phi, align 4
  store i32 %val, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 1
  %B.next = getelementptr inbounds i32, ptr %B.phi, i64 1
  %extra.use = ptrtoint ptr %A.next to i64  ; Extra use of GEP
  call void @use_value(i64 %extra.use)
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Single pointer PHI (should be optimized)
define void @test_single_pointer_phi(ptr %A, ptr %last) {
; CHECK-LABEL: @test_single_pointer_phi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[A_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[A_COMPUTED]], align 4
; CHECK-NEXT:    [[INCREMENTED:%.*]] = add{{.*}} i32 [[VAL]], 1
; CHECK-NEXT:    [[A_PHI_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    store i32 [[INCREMENTED]], ptr [[A_PHI_COMPUTED]], align 4
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_NEXT]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %val = load i32, ptr %A.phi, align 4
  %incremented = add i32 %val, 1
  store i32 %incremented, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 1
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; Negative Cases

; TEST: Base pointer from parent loop PHI
; This is the critical fix - should NOT transform
define void @test_base_from_outer_loop_phi(ptr %start, i32 %outer_n, i32 %inner_n) {
; CHECK-LABEL: @test_base_from_outer_loop_phi(
; CHECK:       outer.header:
; CHECK:         [[OUTER_PTR:%.*]] = phi ptr
; CHECK:       inner.body:
; The inner pointer IV should NOT be converted (base is outer loop PHI)
; CHECK:         [[INNER_PTR:%.*]] = phi ptr
; CHECK:         store i8 0, ptr [[INNER_PTR]]
; CHECK:         [[INNER_NEXT:%.*]] = getelementptr {{.*}} ptr [[INNER_PTR]], i64 1
;
entry:
  %cmp.outer = icmp sgt i32 %outer_n, 0
  br i1 %cmp.outer, label %outer.ph, label %exit

outer.ph:
  %cmp.inner = icmp sgt i32 %inner_n, 0
  br i1 %cmp.inner, label %outer.header, label %exit

outer.header:
  %outer.i = phi i32 [ 0, %outer.ph ], [ %outer.i.next, %inner.exit ]
  %outer.ptr = phi ptr [ %start, %outer.ph ], [ %inner.ptr.lcssa, %inner.exit ]
  br label %inner.body

inner.body:
  %inner.ptr = phi ptr [ %outer.ptr, %outer.header ], [ %inner.ptr.next, %inner.body ]
  %inner.i = phi i32 [ 0, %outer.header ], [ %inner.i.next, %inner.body ]
  store i8 0, ptr %inner.ptr, align 1
  %inner.ptr.next = getelementptr inbounds nuw i8, ptr %inner.ptr, i64 1
  %inner.i.next = add nuw nsw i32 %inner.i, 1
  %cmp.inner.loop = icmp slt i32 %inner.i.next, %inner_n
  br i1 %cmp.inner.loop, label %inner.body, label %inner.exit

inner.exit:
  %inner.ptr.lcssa = phi ptr [ %inner.ptr.next, %inner.body ]
  %outer.i.next = add nuw nsw i32 %outer.i, 1
  %cmp.outer.loop = icmp slt i32 %outer.i.next, %outer_n
  br i1 %cmp.outer.loop, label %outer.header, label %exit

exit:
  ret void
}

; TEST: Base pointer from GEP in parent loop
; Should NOT transform because base is computed in outer loop header
define void @test_base_gep_in_outer_loop(ptr %array, i32 %outer_n, i32 %inner_n) {
; CHECK-LABEL: @test_base_gep_in_outer_loop(
; CHECK:       outer.header:
; CHECK:         [[OUTER_IV:%.*]] = phi i64
; CHECK:         [[BASE_GEP:%.*]] = getelementptr {{.*}} ptr %array, i64 [[OUTER_IV]]
; CHECK:       inner.body:
; The inner pointer IV should NOT be converted (base is GEP from outer loop)
; CHECK:         [[INNER_PTR:%.*]] = phi ptr
; CHECK:         [[INNER_NEXT:%.*]] = getelementptr {{.*}} ptr [[INNER_PTR]], i64 1
;
entry:
  %cmp.outer = icmp sgt i32 %outer_n, 0
  br i1 %cmp.outer, label %outer.ph, label %exit

outer.ph:
  %cmp.inner = icmp sgt i32 %inner_n, 0
  br i1 %cmp.inner, label %outer.header, label %exit

outer.header:
  %outer.iv = phi i64 [ 0, %outer.ph ], [ %outer.iv.next, %inner.exit ]
  ; Base pointer is computed from outer loop IV
  %base.gep = getelementptr inbounds i8, ptr %array, i64 %outer.iv
  br label %inner.body

inner.body:
  %inner.ptr = phi ptr [ %base.gep, %outer.header ], [ %inner.ptr.next, %inner.body ]
  %inner.i = phi i32 [ 0, %outer.header ], [ %inner.i.next, %inner.body ]
  store i8 0, ptr %inner.ptr, align 1
  %inner.ptr.next = getelementptr inbounds nuw i8, ptr %inner.ptr, i64 1
  %inner.i.next = add nuw nsw i32 %inner.i, 1
  %cmp.inner.loop = icmp slt i32 %inner.i.next, %inner_n
  br i1 %cmp.inner.loop, label %inner.body, label %inner.exit

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 64
  %cmp.outer.loop = icmp slt i64 %outer.iv.next, 1024
  br i1 %cmp.outer.loop, label %outer.header, label %exit

exit:
  ret void
}

; TEST: Multiple exit blocks
define void @test_multiple_exits(ptr %base, i32 %n) {
; This should NOT be transformed due to multiple exits
; CHECK-LABEL: @test_multiple_exits(
; CHECK:       loop:
; CHECK:         [[P:%.*]] = phi ptr
; CHECK:         [[P_NEXT:%.*]] = getelementptr
;
entry:
  br label %loop

loop:
  %p = phi ptr [ %base, %entry ], [ %p.next, %loop.latch ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  %val = load i8, ptr %p, align 1
  %cmp.early = icmp eq i8 %val, 0
  br i1 %cmp.early, label %exit.early, label %loop.latch

loop.latch:
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void

exit.early:
  ret void
}

; TEST: Large stride > 64
define void @test_large_stride_rejected(ptr %base, i32 %n) {
; Stride > 64 should NOT be transformed
; CHECK-LABEL: @test_large_stride_rejected(
; CHECK:       loop:
; CHECK:         [[P:%.*]] = phi ptr
; CHECK:         [[P_NEXT:%.*]] = getelementptr {{.*}} ptr [[P]], i64 65
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds i8, ptr %p, i64 65
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Non-constant stride
define void @test_variable_stride_rejected(ptr %base, i32 %n, i64 %stride) {
; Should NOT be transformed - stride is not constant
; CHECK-LABEL: @test_variable_stride_rejected(
; CHECK:       loop:
; CHECK:         [[P:%.*]] = phi ptr
; CHECK:         [[P_NEXT:%.*]] = getelementptr {{.*}} ptr [[P]], i64 %stride
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds i8, ptr %p, i64 %stride
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: GEP with multiple indices
define void @test_multi_index_gep_rejected(ptr %base, i32 %n) {
; Should NOT be transformed - GEP has multiple indices
; CHECK-LABEL: @test_multi_index_gep_rejected(
; CHECK:       loop:
; CHECK:         [[P:%.*]] = phi ptr
; CHECK:         [[P_NEXT:%.*]] = getelementptr {{.*}} ptr [[P]], i64 0, i64 1
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  ; Multi-index GEP
  %p.next = getelementptr inbounds [16 x i8], ptr %p, i64 0, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: GEP pointer operand is not the PHI
define void @test_gep_not_from_phi_rejected(ptr %base, ptr %other, i32 %n) {
; Should NOT be transformed - GEP's pointer operand is not the PHI
; CHECK-LABEL: @test_gep_not_from_phi_rejected(
; CHECK:       loop:
; CHECK:         [[P:%.*]] = phi ptr
; CHECK:         [[P_NEXT:%.*]] = getelementptr {{.*}} ptr %other, i64 1
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  ; GEP from different pointer, not PHI
  %p.next = getelementptr inbounds i8, ptr %other, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Backedge value is not a GEP
define void @test_backedge_not_gep(ptr %base, ptr %alt, i32 %n) {
; Should NOT be transformed - backedge value is not a GEP
; CHECK-LABEL: @test_backedge_not_gep(
; CHECK:       loop:
; CHECK:         [[P:%.*]] = phi ptr [ {{.*}}, {{.*}} ], [ %alt, %loop ]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %alt, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Non-pointer PHI node
; handlePointerIV only handles pointer-type PHIs
; This test verifies that integer PHIs are not affected by handlePointerIV.
define i32 @test_non_pointer_phi_ignored(i32 %start, i32 %n, ptr %out) {
; The integer PHI should not be affected by handlePointerIV
; CHECK-LABEL: @test_non_pointer_phi_ignored(
; CHECK:       loop:
; CHECK:         [[I:%.*]] = phi i32
; CHECK:         store i32 [[I]], ptr %out
; CHECK:         [[I_NEXT:%.*]] = add
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %i = phi i32 [ %start, %loop.ph ], [ %i.next, %loop ]
  ; Use %i to prevent optimization
  store i32 %i, ptr %out, align 4
  %i.next = add nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  ret i32 %result
}

; TEST: Variable stride (should NOT be optimized)
define void @test_variable_stride(ptr %A, ptr %last, ptr %B, i64 %stride) {
; CHECK-LABEL: @test_variable_stride(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP1:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI:%.*]] = phi ptr [ [[A_NEXT:%.*]], [[LOOP1]] ], [ [[A]], [[LOOP]] ]
; CHECK-NEXT:    [[B_PHI:%.*]] = phi ptr [ [[B_NEXT:%.*]], [[LOOP1]] ], [ [[B:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[B_PHI]], align 4
; CHECK-NEXT:    store i32 [[VAL]], ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[A_NEXT]] = getelementptr inbounds i32, ptr [[A_PHI]], i64 [[STRIDE:%.*]]
; CHECK-NEXT:    [[B_NEXT]] = getelementptr inbounds i32, ptr [[B_PHI]], i64 [[STRIDE]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP1]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %val = load i32, ptr %B.phi, align 4
  store i32 %val, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 %stride  ; Variable stride
  %B.next = getelementptr inbounds i32, ptr %B.phi, i64 %stride
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Non-GEP increment (should NOT be optimized)
define void @test_non_gep_increment(ptr %A, ptr %last, i64 %offset) {
; CHECK-LABEL: @test_non_gep_increment(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP1:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI:%.*]] = phi ptr [ [[A_NEXT:%.*]], [[LOOP1]] ], [ [[A]], [[LOOP]] ]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[INCREMENTED:%.*]] = add{{.*}} i32 [[VAL]], 1
; CHECK-NEXT:    store i32 [[INCREMENTED]], ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[A_INT:%.*]] = ptrtoint ptr [[A_PHI]] to i64
; CHECK-NEXT:    [[A_INT_NEXT:%.*]] = add{{.*}} i64 [[A_INT]], [[OFFSET:%.*]]
; CHECK-NEXT:    [[A_NEXT]] = inttoptr i64 [[A_INT_NEXT]] to ptr
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP1]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %val = load i32, ptr %A.phi, align 4
  %incremented = add i32 %val, 1
  store i32 %incremented, ptr %A.phi, align 4
  %A.int = ptrtoint ptr %A.phi to i64
  %A.int.next = add i64 %A.int, %offset  ; Non-GEP increment
  %A.next = inttoptr i64 %A.int.next to ptr
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: GEP that doesn't use PHI as base (should NOT be optimized)
define void @test_gep_wrong_base(ptr %A, ptr %last, ptr %base) {
; CHECK-LABEL: @test_gep_wrong_base(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP1:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI:%.*]] = phi ptr [ [[A_NEXT:%.*]], [[LOOP1]] ], [ [[A]], [[LOOP]] ]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[INCREMENTED:%.*]] = add{{.*}} i32 [[VAL]], 1
; CHECK-NEXT:    store i32 [[INCREMENTED]], ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[A_NEXT]] = getelementptr inbounds i32, ptr [[BASE:%.*]], i64 1
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP1]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %val = load i32, ptr %A.phi, align 4
  %incremented = add i32 %val, 1
  store i32 %incremented, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %base, i64 1  ; Wrong base pointer
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: PHI that has more than 2 incoming values (should NOT be optimized)
define void @test_complex_phi(ptr %A, ptr %last, i1 %cond) {
; CHECK-LABEL: @test_complex_phi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI:%.*]] = phi ptr [ [[A]], [[LOOP_PREHEADER]] ], [ [[A_PHI_BE:%.*]], [[LOOP_BACKEDGE:%.*]] ]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[INCREMENTED:%.*]] = add{{.*}} i32 [[VAL]], 1
; CHECK-NEXT:    store i32 [[INCREMENTED]], ptr [[A_PHI]], align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[LOOP_ALT:%.*]], label [[LOOP_BACK:%.*]]
; CHECK:       loop.alt:
; CHECK-NEXT:    [[A_ALT:%.*]] = getelementptr inbounds i32, ptr [[A_PHI]], i64 2
; CHECK-NEXT:    br label [[LOOP_BACKEDGE]]
; CHECK:       loop.backedge:
; CHECK-NEXT:    [[A_PHI_BE]] = phi ptr [ [[A_NEXT:%.*]], [[LOOP_BACK]] ], [ [[A_ALT]], [[LOOP_ALT]] ]
; CHECK-NEXT:    br label [[LOOP]]
; CHECK:       loop.back:
; CHECK-NEXT:    [[A_NEXT]] = getelementptr inbounds i32, ptr [[A_PHI]], i64 1
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP_BACKEDGE]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop.preheader

loop.preheader:
  br label %loop

loop:
  %A.phi = phi ptr [ %A, %loop.preheader ], [ %A.next, %loop.back ], [ %A.alt, %loop.alt ]
  %val = load i32, ptr %A.phi, align 4
  %incremented = add i32 %val, 1
  store i32 %incremented, ptr %A.phi, align 4
  br i1 %cond, label %loop.alt, label %loop.back

loop.alt:
  %A.alt = getelementptr inbounds i32, ptr %A.phi, i64 2
  br label %loop

loop.back:
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 1
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Zero stride (should NOT be optimized)
define void @test_zero_stride(ptr %A, ptr %last) {
; CHECK-LABEL: @test_zero_stride(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A_PHI:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP1:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[INCREMENTED:%.*]] = add{{.*}} i32 [[VAL]], 1
; CHECK-NEXT:    store i32 [[INCREMENTED]], ptr [[A_PHI]], align 4
; CHECK-NEXT:    br i1 false, label [[EXIT_LOOPEXIT:%.*]], label [[LOOP1]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %val = load i32, ptr %A.phi, align 4
  %incremented = add i32 %val, 1
  store i32 %incremented, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 0  ; Zero stride
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Non-pointer PHI (should be ignored)
define void @test_non_pointer_phi(ptr %A, ptr %last, i32 %start) {
; CHECK-LABEL: @test_non_pointer_phi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP1:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[I_PHI:%.*]] = phi i32 [ [[I_NEXT:%.*]], [[LOOP1]] ], [ [[START:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_GEP:%.*]] = getelementptr inbounds i32, ptr [[A]], i32 [[I_PHI]]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[A_GEP]], align 4
; CHECK-NEXT:    [[INCREMENTED:%.*]] = add{{.*}} i32 [[VAL]], 1
; CHECK-NEXT:    store i32 [[INCREMENTED]], ptr [[A_GEP]], align 4
; CHECK-NEXT:    [[I_NEXT]] = add{{.*}} i32 [[I_PHI]], 1
; CHECK-NEXT:    [[A_NEXT:%.*]] = getelementptr inbounds i32, ptr [[A]], i32 [[I_NEXT]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP1]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %i.phi = phi i32 [ %i.next, %loop ], [ %start, %entry ]  ; Integer PHI, not pointer
  %A.gep = getelementptr inbounds i32, ptr %A, i32 %i.phi
  %val = load i32, ptr %A.gep, align 4
  %incremented = add i32 %val, 1
  store i32 %incremented, ptr %A.gep, align 4
  %i.next = add i32 %i.phi, 1
  %A.next = getelementptr inbounds i32, ptr %A, i32 %i.next
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; Edge Cases

; TEST: LCSSA phi skip
; When the increment GEP has a use outside the loop (LCSSA phi), the patch
; skips creating a computed GEP for it, letting SCEV handle exit values.
; SCEV may still compute the exit value and replace the LCSSA phi.
define ptr @test_lcssa_phi_skip(ptr %base, i32 %n) {
; CHECK-LABEL: @test_lcssa_phi_skip(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[IV_NEXT:%.*]] = add nuw nsw i64 [[IV]], 1
; CHECK:         [[PTR:%.*]] = getelementptr {{.*}} ptr %base, i64 [[IV]]
; CHECK:         store i8 42, ptr [[PTR]]
; CHECK:       exit:
; SCEV computes exit value - either a phi or computed scevgep
; CHECK:         ret ptr
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit.early

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ; LCSSA phi - the value escapes the loop
  %p.lcssa = phi ptr [ %p.next, %loop ]
  ret ptr %p.lcssa

exit.early:
  ret ptr null
}

; TEST: Pointer PHI with external use
; The pointer PHI itself has a use outside the loop (an LCSSA phi), so it is
; kept live (like Incr's outside-loop uses below) instead of being eagerly
; rewritten with a computed GEP.
define ptr @test_phi_with_external_use(ptr %base, i32 %n) {
; CHECK-LABEL: @test_phi_with_external_use(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[P:%.*]] = phi ptr [ %base, {{.*}} ], [ %p.next, %loop ]
; CHECK:         [[PTR:%.*]] = getelementptr {{.*}} ptr %base, i64 [[IV]]
; CHECK:       exit:
; CHECK:         [[P_EXIT:%.*]] = phi ptr [ [[P]], %loop ]
; CHECK:         ret ptr [[P_EXIT]]
;
entry:
  br label %loop.ph

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 42, ptr %p, align 1
  %p.next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret ptr %p
}

; TEST: Boundary stride value -64
define void @test_stride_minus_64(ptr %end, i32 %n) {
; Stride -64 is within bounds (abs <= 64) - SHOULD be transformed
; CHECK-LABEL: @test_stride_minus_64(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[SCALED:%.*]] = mul {{.*}} i64 [[IV]], -64
; CHECK:         [[PTR:%.*]] = getelementptr {{.*}} ptr %end, i64 [[SCALED]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %end, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 0, ptr %p, align 1
  %p.next = getelementptr inbounds i8, ptr %p, i64 -64
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Boundary stride value +64
define void @test_stride_plus_64(ptr %base, i32 %n) {
; Stride +64 is within bounds (abs <= 64) - SHOULD be transformed
; CHECK-LABEL: @test_stride_plus_64(
; CHECK:       loop:
; CHECK:         [[IV:%.*]] = phi i64
; CHECK:         [[SCALED:%.*]] = mul {{.*}} i64 [[IV]], 64
; CHECK:         [[PTR:%.*]] = getelementptr {{.*}} ptr %base, i64 [[SCALED]]
;
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %p = phi ptr [ %base, %loop.ph ], [ %p.next, %loop ]
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  store i8 0, ptr %p, align 1
  %p.next = getelementptr inbounds i8, ptr %p, i64 64
  %i.next = add nuw nsw i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %n
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

; TEST: Maximum allowed stride (should be optimized)
define void @test_large_stride(ptr %A, ptr %last) {
; CHECK-LABEL: @test_large_stride(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP1:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP1]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[A_PHI_INT_SCALED1:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], 64
; CHECK-NEXT:    [[A_PHI:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_SCALED1]]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[A_PHI]], align 4
; CHECK-NEXT:    [[INCREMENTED:%.*]] = add{{.*}} i32 [[VAL]], 1
; CHECK-NEXT:    [[A_PHI_INT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], 64
; CHECK-NEXT:    [[A_PHI_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_SCALED]]
; CHECK-NEXT:    store i32 [[INCREMENTED]], ptr [[A_PHI_COMPUTED]], align 4
; CHECK-NEXT:    [[A_PHI_INT_NEXT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT_NEXT]], 64
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_NEXT_SCALED]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP1]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %val = load i32, ptr %A.phi, align 4
  %incremented = add i32 %val, 1
  store i32 %incremented, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 64  ; Maximum allowed stride
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Nested loop pointer IV with outer-loop pointer base
; This is a reduced test case from Perl_repeatcpy in SPEC CPU2006 perlbench.
; The inner loop copies 'len' bytes from 'from' to 'to', and this is repeated
; 'count' times. The 'to' pointer accumulates across outer iterations.
;
; With the fix, pointer IVs whose base is from an outer loop are NOT transformed
; to integer IVs, while pointer IVs with loop-invariant bases (like function
; parameters) are still transformed.
define void @test_nested_loop_ptr_iv(ptr %to, ptr %from, i32 %len, i32 %count) {
; CHECK-LABEL: @test_nested_loop_ptr_iv(
; CHECK:       outer.header:
; CHECK-NEXT:    [[OUTER_COUNT:%.*]] = phi i32
; CHECK-NEXT:    [[TO_OUTER:%.*]] = phi ptr
; CHECK:       inner.body:
; The to.inner pointer IV should NOT be converted (base is outer loop IV)
; CHECK:         [[TO_INNER:%.*]] = phi ptr
; The from.inner pointer IV SHOULD be converted to integer IV (base is function param)
; CHECK:         [[FROM_IV:%.*]] = phi i64
; CHECK:         [[FROM_COMPUTED:%.*]] = getelementptr {{.*}} ptr [[FROM:%.*]], i64 [[FROM_IV]]
; CHECK:         load i8, ptr [[FROM_COMPUTED]]
; The to pointer increment should be kept as a GEP
; CHECK:         [[TO_NEXT:%.*]] = getelementptr {{.*}} ptr [[TO_INNER]], i64 1
; CHECK:         store i8 {{.*}}, ptr [[TO_INNER]]
; CHECK:       inner.exit:
; The LCSSA phi should be preserved for the to pointer
; CHECK:         [[TO_LCSSA:%.*]] = phi ptr
; CHECK:         br
;
entry:
  %cmp.outer.guard = icmp sgt i32 %count, 0
  br i1 %cmp.outer.guard, label %outer.preheader, label %exit

outer.preheader:
  %cmp.inner.guard = icmp sgt i32 %len, 0
  br i1 %cmp.inner.guard, label %outer.header, label %exit

outer.header:
  %outer.count = phi i32 [ %count, %outer.preheader ], [ %outer.count.dec, %inner.exit ]
  %to.outer = phi ptr [ %to, %outer.preheader ], [ %to.lcssa, %inner.exit ]
  %outer.count.dec = add nsw i32 %outer.count, -1
  br label %inner.body

inner.body:
  %to.inner = phi ptr [ %to.inner.next, %inner.body ], [ %to.outer, %outer.header ]
  %from.inner = phi ptr [ %from.inner.next, %inner.body ], [ %from, %outer.header ]
  %inner.count = phi i32 [ %inner.count.dec, %inner.body ], [ %len, %outer.header ]
  %from.inner.next = getelementptr inbounds nuw i8, ptr %from.inner, i64 1
  %val = load i8, ptr %from.inner, align 1
  %to.inner.next = getelementptr inbounds nuw i8, ptr %to.inner, i64 1
  store i8 %val, ptr %to.inner, align 1
  %inner.count.dec = add nsw i32 %inner.count, -1
  %cmp.inner = icmp sgt i32 %inner.count, 1
  br i1 %cmp.inner, label %inner.body, label %inner.exit

inner.exit:
  ; LCSSA phi - this captures the final value of to.inner.next
  ; which should be to.outer + len
  %to.lcssa = phi ptr [ %to.inner.next, %inner.body ]
  %cmp.outer = icmp sgt i32 %outer.count, 1
  br i1 %cmp.outer, label %outer.header, label %exit

exit:
  ret void
}

; TEST: Nested memcpy pattern with inner pointer IV from outer loop
; Both loops use pointer IVs, and the inner loop's destination base comes
; from the outer loop.
; Verify memory access pattern is preserved.
define void @test_nested_memcpy_pattern(ptr %dst, ptr %src, i32 %chunk_size, i32 %num_chunks) {
; CHECK-LABEL: @test_nested_memcpy_pattern(
; CHECK:       outer.loop:
; CHECK:         [[DST_OUTER:%.*]] = phi ptr
; CHECK:       inner.loop:
; The dst.inner pointer IV should NOT be converted (base is outer loop IV)
; CHECK:         [[DST_INNER:%.*]] = phi ptr
; The source should be computed using integer IV from the original 'src' base
; CHECK:         [[SRC_IV:%.*]] = phi i64
; CHECK:         [[SRC_ADDR:%.*]] = getelementptr {{.*}} ptr [[SRC:%.*]], i64 [[SRC_IV]]
; The destination increment should be kept as a GEP
; CHECK:         [[DST_INC:%.*]] = getelementptr {{.*}} ptr [[DST_INNER]], i64 1
;
entry:
  %has_chunks = icmp sgt i32 %num_chunks, 0
  br i1 %has_chunks, label %check.size, label %done

check.size:
  %has_data = icmp sgt i32 %chunk_size, 0
  br i1 %has_data, label %outer.loop, label %done

outer.loop:
  %chunks.remaining = phi i32 [ %num_chunks, %check.size ], [ %chunks.dec, %inner.done ]
  %dst.outer = phi ptr [ %dst, %check.size ], [ %dst.next, %inner.done ]
  %chunks.dec = add nsw i32 %chunks.remaining, -1
  br label %inner.loop

inner.loop:
  %dst.inner = phi ptr [ %dst.inner.inc, %inner.loop ], [ %dst.outer, %outer.loop ]
  %src.inner = phi ptr [ %src.inner.inc, %inner.loop ], [ %src, %outer.loop ]
  %bytes.left = phi i32 [ %bytes.dec, %inner.loop ], [ %chunk_size, %outer.loop ]
  %src.inner.inc = getelementptr inbounds nuw i8, ptr %src.inner, i64 1
  %byte = load i8, ptr %src.inner, align 1
  %dst.inner.inc = getelementptr inbounds nuw i8, ptr %dst.inner, i64 1
  store i8 %byte, ptr %dst.inner, align 1
  %bytes.dec = add nsw i32 %bytes.left, -1
  %more.bytes = icmp sgt i32 %bytes.left, 1
  br i1 %more.bytes, label %inner.loop, label %inner.done

inner.done:
  %dst.next = phi ptr [ %dst.inner.inc, %inner.loop ]
  %more.chunks = icmp sgt i32 %chunks.remaining, 1
  br i1 %more.chunks, label %outer.loop, label %done

done:
  ret void
}

; TEST: GEP-derived base from outer loop IV
; The SCEV-based check should detect this case (BasePtr is SCEVAddRecExpr in parent loop).
; This tests the fix for the critical issue where we only checked for PHI nodes.
define void @test_gep_derived_base_from_outer_iv(ptr %base, i32 %outer_iters, i32 %inner_iters) {
; CHECK-LABEL: @test_gep_derived_base_from_outer_iv(
; CHECK:       outer.header:
; CHECK:         [[OUTER_IV:%.*]] = phi i64
; CHECK:         [[OUTER_PTR:%.*]] = getelementptr {{.*}} ptr %base, i64 [[OUTER_IV]]
; At the start of the inner loop, we add an offset to the outer pointer.
; This derived pointer (base + outer_iv + offset) should NOT have its
; inner loop IV transformed because it varies with the outer loop.
; CHECK:         [[INNER_BASE:%.*]] = getelementptr {{.*}} ptr [[OUTER_PTR]], i64 16
; CHECK:       inner.body:
; The inner pointer IV should NOT be converted (base derives from outer loop IV)
; CHECK:         [[INNER_PTR:%.*]] = phi ptr
; CHECK:         [[INNER_NEXT:%.*]] = getelementptr {{.*}} ptr [[INNER_PTR]], i64 1
; CHECK:         store i8 {{.*}}, ptr [[INNER_PTR]]
; CHECK:       inner.exit:
; CHECK:         [[INNER_LCSSA:%.*]] = phi ptr
;
entry:
  %cmp.outer = icmp sgt i32 %outer_iters, 0
  br i1 %cmp.outer, label %outer.preheader, label %exit

outer.preheader:
  %cmp.inner = icmp sgt i32 %inner_iters, 0
  br i1 %cmp.inner, label %outer.header, label %exit

outer.header:
  %outer.iv = phi i64 [ 0, %outer.preheader ], [ %outer.iv.next, %inner.exit ]
  ; Outer loop IV is used to compute a pointer
  %outer.ptr = getelementptr inbounds i8, ptr %base, i64 %outer.iv
  ; Inner loop base is derived from the outer loop IV via GEP
  %inner.base = getelementptr inbounds i8, ptr %outer.ptr, i64 16
  br label %inner.body

inner.body:
  %inner.ptr = phi ptr [ %inner.base, %outer.header ], [ %inner.next, %inner.body ]
  %inner.count = phi i32 [ %inner_iters, %outer.header ], [ %inner.count.dec, %inner.body ]
  %inner.next = getelementptr inbounds nuw i8, ptr %inner.ptr, i64 1
  store i8 42, ptr %inner.ptr, align 1
  %inner.count.dec = add nsw i32 %inner.count, -1
  %inner.cmp = icmp sgt i32 %inner.count, 1
  br i1 %inner.cmp, label %inner.body, label %inner.exit

inner.exit:
  %inner.lcssa = phi ptr [ %inner.next, %inner.body ]
  %outer.iv.next = add nuw nsw i64 %outer.iv, 32
  %outer.count.cmp = icmp slt i64 %outer.iv.next, 256
  br i1 %outer.count.cmp, label %outer.header, label %exit

exit:
  ret void
}

; Other Coverage

; TEST: Basic pointer PHI elimination with int pointers
define void @test_basic_int_ptr(ptr %A, ptr %last, ptr %B, ptr %C) {
; CHECK-LABEL: @test_basic_int_ptr(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[C_COMPUTED:%.*]] = getelementptr i32, ptr [[B:%.*]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    [[VAL_B:%.*]] = load i32, ptr [[C_COMPUTED]], align 4
; CHECK-NEXT:    [[C_PHI_COMPUTED:%.*]] = getelementptr i32, ptr [[C:%.*]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    [[VAL_C:%.*]] = load i32, ptr [[C_PHI_COMPUTED]], align 4
; CHECK-NEXT:    [[SUM:%.*]] = add nsw i32 [[VAL_B]], [[VAL_C]]
; CHECK-NEXT:    [[A_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    store i32 [[SUM]], ptr [[A_COMPUTED]], align 4
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_NEXT]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %C.phi = phi ptr [ %C.next, %loop ], [ %C, %entry ]
  %val.B = load i32, ptr %B.phi, align 4
  %val.C = load i32, ptr %C.phi, align 4
  %sum = add nsw i32 %val.B, %val.C
  store i32 %sum, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 1
  %B.next = getelementptr inbounds i32, ptr %B.phi, i64 1
  %C.next = getelementptr inbounds i32, ptr %C.phi, i64 1
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Different data types char pointers
define void @test_char_ptr(ptr %A, ptr %last, ptr %B) {
; CHECK-LABEL: @test_char_ptr(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[B_COMPUTED:%.*]] = getelementptr i8, ptr [[B:%.*]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    [[VAL:%.*]] = load i8, ptr [[B_COMPUTED]], align 1
; CHECK-NEXT:    [[A_COMPUTED:%.*]] = getelementptr i8, ptr [[A]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    store i8 [[VAL]], ptr [[A_COMPUTED]], align 1
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i8, ptr [[A]], i64 [[A_PHI_INT_NEXT]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %val = load i8, ptr %B.phi, align 1
  store i8 %val, ptr %A.phi, align 1
  %A.next = getelementptr inbounds i8, ptr %A.phi, i64 1
  %B.next = getelementptr inbounds i8, ptr %B.phi, i64 1
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Struct pointers
define void @test_struct_ptr(ptr %A, ptr %last, ptr %B) {
; CHECK-LABEL: @test_struct_ptr(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[LAST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[B_COMPUTED:%.*]] = getelementptr [[STRUCT_POINT:%.*]], ptr [[B:%.*]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    [[VAL:%.*]] = load [[STRUCT_POINT]], ptr [[B_COMPUTED]], align 4
; CHECK-NEXT:    [[A_COMPUTED:%.*]] = getelementptr [[STRUCT_POINT]], ptr [[A]], i64 [[A_PHI_INT]]
; CHECK-NEXT:    store [[STRUCT_POINT]] [[VAL]], ptr [[A_COMPUTED]], align 4
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr [[STRUCT_POINT]], ptr [[A]], i64 [[A_PHI_INT_NEXT]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[LAST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %last
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %val = load %struct.Point, ptr %B.phi, align 4
  store %struct.Point %val, ptr %A.phi, align 4
  %A.next = getelementptr inbounds %struct.Point, ptr %A.phi, i64 1
  %B.next = getelementptr inbounds %struct.Point, ptr %B.phi, i64 1
  %cmp.next = icmp eq ptr %A.next, %last
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Mixed pointer types in same loop (should optimize each independently)
define void @test_mixed_types(ptr %A_int, ptr %A_char, ptr %last_int, ptr %last_char, ptr %B_int, ptr %B_char) {
; CHECK-LABEL: @test_mixed_types(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq ptr [[A_INT:%.*]], [[LAST_INT:%.*]]
; CHECK-NEXT:    [[CMP2:%.*]] = icmp eq ptr [[A_CHAR:%.*]], [[LAST_CHAR:%.*]]
; CHECK-NEXT:    [[CMP:%.*]] = or i1 [[CMP1]], [[CMP2]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_INT_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_INT_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_INT_PHI_INT_NEXT]] = add{{.*}} i64 [[A_INT_PHI_INT]], 1
; CHECK-NEXT:    [[B_INT_COMPUTED:%.*]] = getelementptr i32, ptr [[B_INT:%.*]], i64 [[A_INT_PHI_INT]]
; CHECK-NEXT:    [[VAL_INT:%.*]] = load i32, ptr [[B_INT_COMPUTED]], align 4
; CHECK-NEXT:    [[B_CHAR_COMPUTED:%.*]] = getelementptr i8, ptr [[B_CHAR:%.*]], i64 [[A_INT_PHI_INT]]
; CHECK-NEXT:    [[VAL_CHAR:%.*]] = load i8, ptr [[B_CHAR_COMPUTED]], align 1
; CHECK-NEXT:    [[A_INT_COMPUTED:%.*]] = getelementptr i32, ptr [[A_INT]], i64 [[A_INT_PHI_INT]]
; CHECK-NEXT:    store i32 [[VAL_INT]], ptr [[A_INT_COMPUTED]], align 4
; CHECK-NEXT:    [[A_CHAR_COMPUTED:%.*]] = getelementptr i8, ptr [[A_CHAR]], i64 [[A_INT_PHI_INT]]
; CHECK-NEXT:    store i8 [[VAL_CHAR]], ptr [[A_CHAR_COMPUTED]], align 1
; CHECK-NEXT:    [[A_INT_NEXT_COMPUTED:%.*]] = getelementptr i32, ptr [[A_INT]], i64 [[A_INT_PHI_INT_NEXT]]
; CHECK-NEXT:    [[CMP1_NEXT:%.*]] = icmp eq ptr [[A_INT_NEXT_COMPUTED]], [[LAST_INT]]
; CHECK-NEXT:    [[A_CHAR_NEXT_COMPUTED:%.*]] = getelementptr i8, ptr [[A_CHAR]], i64 [[A_INT_PHI_INT_NEXT]]
; CHECK-NEXT:    [[CMP2_NEXT:%.*]] = icmp eq ptr [[A_CHAR_NEXT_COMPUTED]], [[LAST_CHAR]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = or i1 [[CMP1_NEXT]], [[CMP2_NEXT]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp1 = icmp eq ptr %A_int, %last_int
  %cmp2 = icmp eq ptr %A_char, %last_char
  %cmp = or i1 %cmp1, %cmp2
  br i1 %cmp, label %exit, label %loop

loop:
  %A_int.phi = phi ptr [ %A_int.next, %loop ], [ %A_int, %entry ]
  %A_char.phi = phi ptr [ %A_char.next, %loop ], [ %A_char, %entry ]
  %B_int.phi = phi ptr [ %B_int.next, %loop ], [ %B_int, %entry ]
  %B_char.phi = phi ptr [ %B_char.next, %loop ], [ %B_char, %entry ]
  %val.int = load i32, ptr %B_int.phi, align 4
  %val.char = load i8, ptr %B_char.phi, align 1
  store i32 %val.int, ptr %A_int.phi, align 4
  store i8 %val.char, ptr %A_char.phi, align 1
  %A_int.next = getelementptr inbounds i32, ptr %A_int.phi, i64 1
  %A_char.next = getelementptr inbounds i8, ptr %A_char.phi, i64 1
  %B_int.next = getelementptr inbounds i32, ptr %B_int.phi, i64 1
  %B_char.next = getelementptr inbounds i8, ptr %B_char.phi, i64 1
  %cmp1.next = icmp eq ptr %A_int.next, %last_int
  %cmp2.next = icmp eq ptr %A_char.next, %last_char
  %cmp.next = or i1 %cmp1.next, %cmp2.next
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Pointer IV transformation fix
define void @test_pointer_iv_transformation(ptr %p2.2, ptr %q.4, ptr %p.4) {
; CHECK-LABEL: define void @test_pointer_iv_transformation(
; CHECK-SAME: ptr [[P2_2:%.*]], ptr [[Q_4:%.*]], ptr [[P_4:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    br label %[[WHILE_BODY81_PREHEADER218:.*]]
; CHECK:       [[WHILE_BODY81_PREHEADER218]]:
; CHECK-NEXT:    br label %[[WHILE_BODY81:.*]]
; CHECK:       [[WHILE_BODY81]]:
; CHECK-NEXT:    [[RUNS_3193:%.*]] = phi i64 [ [[INC83:%.*]], %[[WHILE_BODY81]] ], [ 0, %[[WHILE_BODY81_PREHEADER218]] ]
; CHECK-NEXT:    [[INC83]] = add{{.*}} i64 [[RUNS_3193]], 1
; CHECK-NEXT:    [[P2_3192_INT_SCALED:%.*]] = mul{{.*}} i64 [[RUNS_3193]], 2
; CHECK-NEXT:    [[P2_3192_COMPUTED:%.*]] = getelementptr ptr, ptr [[P2_2]], i64 [[P2_3192_INT_SCALED]]
; CHECK-NEXT:    [[P2_3192_INT_NEXT_SCALED:%.*]] = mul{{.*}} i64 [[INC83]], 2
; CHECK-NEXT:    [[ADD_PTR82_COMPUTED:%.*]] = getelementptr ptr, ptr [[P2_2]], i64 [[P2_3192_INT_NEXT_SCALED]]
; CHECK-NEXT:    store ptr [[ADD_PTR82_COMPUTED]], ptr [[P2_3192_COMPUTED]], align 8
; CHECK-NEXT:    [[Q_5191_INT_NEXT_SCALED:%.*]] = mul{{.*}} i64 [[INC83]], 2
; CHECK-NEXT:    [[Q_6_COMPUTED:%.*]] = getelementptr ptr, ptr [[Q_4]], i64 [[Q_5191_INT_NEXT_SCALED]]
; CHECK-NEXT:    [[CMP79:%.*]] = icmp ult ptr [[Q_6_COMPUTED]], [[P_4]]
; CHECK-NEXT:    br i1 [[CMP79]], label %[[WHILE_BODY81]], label %[[WHILE_END93_LOOPEXIT219:.*]]
; CHECK:       [[WHILE_END93_LOOPEXIT219]]:
; CHECK-NEXT:    ret void
;
entry:
  br label %while.body81.preheader218

while.body81.preheader218:
  br label %while.body81

while.body81:                                     ; preds = %while.body81.preheader218, %while.body81
  %runs.3193 = phi i64 [ %inc83, %while.body81 ], [ 0, %while.body81.preheader218 ]
  %p2.3192 = phi ptr [ %add.ptr82, %while.body81 ], [ %p2.2, %while.body81.preheader218 ]
  %q.5191 = phi ptr [ %q.6, %while.body81 ], [ %q.4, %while.body81.preheader218 ]
  %add.ptr82 = getelementptr inbounds nuw ptr, ptr %p2.3192, i64 2
  store ptr %add.ptr82, ptr %p2.3192, align 8
  %inc83 = add nsw i64 %runs.3193, 1
  %q.6 = getelementptr inbounds nuw ptr, ptr %q.5191, i64 2
  %cmp79 = icmp ult ptr %q.6, %p.4
  br i1 %cmp79, label %while.body81, label %while.end93.loopexit219

while.end93.loopexit219:
  ret void
}

; TEST: Basic negative stride with int pointers
define void @test_negative_stride_int(ptr %A, ptr %first, ptr %B) {
; CHECK-LABEL: @test_negative_stride_int(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[FIRST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[B_PHI_INT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], -1
; CHECK-NEXT:    [[B_COMPUTED:%.*]] = getelementptr i32, ptr [[B:%.*]], i64 [[B_PHI_INT_SCALED]]
; CHECK-NEXT:    [[VAL:%.*]] = load i32, ptr [[B_COMPUTED]], align 4
; CHECK-NEXT:    [[A_PHI_INT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], -1
; CHECK-NEXT:    [[A_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_SCALED]]
; CHECK-NEXT:    store i32 [[VAL]], ptr [[A_COMPUTED]], align 4
; CHECK-NEXT:    [[A_PHI_INT_NEXT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT_NEXT]], -1
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i32, ptr [[A]], i64 [[A_PHI_INT_NEXT_SCALED]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[FIRST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %first
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %val = load i32, ptr %B.phi, align 4
  store i32 %val, ptr %A.phi, align 4
  %A.next = getelementptr inbounds i32, ptr %A.phi, i64 -1
  %B.next = getelementptr inbounds i32, ptr %B.phi, i64 -1
  %cmp.next = icmp eq ptr %A.next, %first
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}

; TEST: Negative stride with char pointers
define void @test_negative_stride_char(ptr %A, ptr %first, ptr %B) {
; CHECK-LABEL: @test_negative_stride_char(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[A:%.*]], [[FIRST:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP_PREHEADER:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[A_PHI_INT:%.*]] = phi i64 [ 0, [[LOOP_PREHEADER]] ], [ [[A_PHI_INT_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[A_PHI_INT_NEXT]] = add{{.*}} i64 [[A_PHI_INT]], 1
; CHECK-NEXT:    [[B_PHI_INT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], -1
; CHECK-NEXT:    [[B_COMPUTED:%.*]] = getelementptr i8, ptr [[B:%.*]], i64 [[B_PHI_INT_SCALED]]
; CHECK-NEXT:    [[VAL:%.*]] = load i8, ptr [[B_COMPUTED]], align 1
; CHECK-NEXT:    [[A_PHI_INT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT]], -1
; CHECK-NEXT:    [[A_COMPUTED:%.*]] = getelementptr i8, ptr [[A]], i64 [[A_PHI_INT_SCALED]]
; CHECK-NEXT:    store i8 [[VAL]], ptr [[A_COMPUTED]], align 1
; CHECK-NEXT:    [[A_PHI_INT_NEXT_SCALED:%.*]] = mul{{.*}} i64 [[A_PHI_INT_NEXT]], -1
; CHECK-NEXT:    [[A_NEXT_COMPUTED:%.*]] = getelementptr i8, ptr [[A]], i64 [[A_PHI_INT_NEXT_SCALED]]
; CHECK-NEXT:    [[CMP_NEXT:%.*]] = icmp eq ptr [[A_NEXT_COMPUTED]], [[FIRST]]
; CHECK-NEXT:    br i1 [[CMP_NEXT]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP]]
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  %cmp = icmp eq ptr %A, %first
  br i1 %cmp, label %exit, label %loop

loop:
  %A.phi = phi ptr [ %A.next, %loop ], [ %A, %entry ]
  %B.phi = phi ptr [ %B.next, %loop ], [ %B, %entry ]
  %val = load i8, ptr %B.phi, align 1
  store i8 %val, ptr %A.phi, align 1
  %A.next = getelementptr inbounds i8, ptr %A.phi, i64 -1
  %B.next = getelementptr inbounds i8, ptr %B.phi, i64 -1
  %cmp.next = icmp eq ptr %A.next, %first
  br i1 %cmp.next, label %exit, label %loop

exit:
  ret void
}
