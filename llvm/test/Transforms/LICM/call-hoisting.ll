; RUN: opt -S -licm %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s

declare i32 @load(ptr %p) argmemonly readonly nounwind

define void @test_load(ptr noalias %loc, ptr noalias %sink) {
; CHECK-LABEL: @test_load
; CHECK-LABEL: entry:
; CHECK: call i32 @load
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %ret = call i32 @load(ptr %loc)
  store volatile i32 %ret, ptr %sink
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare i32 @spec(ptr %p, ptr %q) readonly argmemonly nounwind speculatable

; We should strip the dereferenceable callsite attribute on spec call's argument since it is
; can cause UB in the speculatable call when hoisted to preheader.
; However, we need not strip the nonnull attribute since it just propagates
; poison if the parameter was indeed null.
define void @test_strip_attribute(ptr noalias %loc, ptr noalias %sink, ptr %q) {
; CHECK-LABEL: @test_strip_attribute(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[RET:%.*]] = call i32 @load(ptr [[LOC:%.*]])
; CHECK-NEXT:    [[NULLCHK:%.*]] = icmp eq ptr [[Q:%.*]], null
; CHECK-NEXT:    [[RET2:%.*]] = call i32 @spec(ptr nonnull [[Q]], ptr [[LOC]])
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[ISNULL:%.*]] ]
; CHECK-NEXT:    br i1 [[NULLCHK]], label [[ISNULL]], label [[NONNULLBB:%.*]]
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %isnull ]
  %ret = call i32 @load(ptr %loc)
  %nullchk = icmp eq ptr %q, null
  br i1 %nullchk, label %isnull, label %nonnullbb

nonnullbb:
  %ret2 = call i32 @spec(ptr nonnull %q, ptr dereferenceable(12) %loc)
  br label %isnull

isnull:
  store volatile i32 %ret, ptr %sink
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @store(i32 %val, ptr %p) argmemonly writeonly nounwind

define void @test(ptr %loc) {
; CHECK-LABEL: @test
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, ptr %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @test_multiexit(ptr %loc, i1 %earlycnd) {
; CHECK-LABEL: @test_multiexit
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: backedge:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  call void @store(i32 0, ptr %loc)
  %iv.next = add i32 %iv, 1
  br i1 %earlycnd, label %exit1, label %backedge

backedge:
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit2

exit1:
  ret void
exit2:
  ret void
}

define void @neg_lv_value(ptr %loc) {
; CHECK-LABEL: @neg_lv_value
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 %iv, ptr %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_lv_addr(ptr %loc) {
; CHECK-LABEL: @neg_lv_addr
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %p = getelementptr i32, ptr %loc, i32 %iv
  call void @store(i32 0, ptr %p)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_mod(ptr %loc) {
; CHECK-LABEL: @neg_mod
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, ptr %loc)
  store i32 %iv, ptr %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_ref(ptr %loc) {
; CHECK-LABEL: @neg_ref
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit1:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  call void @store(i32 0, ptr %loc)
  %v = load i32, ptr %loc
  %earlycnd = icmp eq i32 %v, 198
  br i1 %earlycnd, label %exit1, label %backedge

backedge:
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit2

exit1:
  ret void
exit2:
  ret void
}

declare void @modref()

define void @neg_modref(ptr %loc) {
; CHECK-LABEL: @neg_modref
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, ptr %loc)
  call void @modref()
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_fence(ptr %loc) {
; CHECK-LABEL: @neg_fence
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, ptr %loc)
  fence seq_cst
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @not_nounwind(i32 %v, ptr %p) writeonly argmemonly
declare void @not_argmemonly(i32 %v, ptr %p) writeonly nounwind
declare void @not_writeonly(i32 %v, ptr %p) argmemonly nounwind

define void @neg_not_nounwind(ptr %loc) {
; CHECK-LABEL: @neg_not_nounwind
; CHECK-LABEL: loop:
; CHECK: call void @not_nounwind
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @not_nounwind(i32 0, ptr %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_not_argmemonly(ptr %loc) {
; CHECK-LABEL: @neg_not_argmemonly
; CHECK-LABEL: loop:
; CHECK: call void @not_argmemonly
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @not_argmemonly(i32 0, ptr %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_not_writeonly(ptr %loc) {
; CHECK-LABEL: @neg_not_writeonly
; CHECK-LABEL: loop:
; CHECK: call void @not_writeonly
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @not_writeonly(i32 0, ptr %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

