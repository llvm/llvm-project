; RUN: opt -S -basic-aa -licm %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s

declare i32 @load(i32* %p) argmemonly readonly nounwind

define void @test_load(i32* noalias %loc, i32* noalias %sink) {
; CHECK-LABEL: @test_load
; CHECK-LABEL: entry:
; CHECK: call i32 @load
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %ret = call i32 @load(i32* %loc)
  store volatile i32 %ret, i32* %sink
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare i32 @spec(i32* %p) readonly argmemonly nounwind speculatable

; FIXME: We should strip the nonnull callsite attribute on spec call's argument since it is
; may not be valid when hoisted to preheader.
define void @test_strip_attribute(i32* noalias %loc, i32* noalias %sink, i32* %q) {
; CHECK-LABEL: test_strip_attribute
; CHECK-LABEL: entry
; CHECK-NEXT:   %ret = call i32 @load(i32* %loc)
; CHECK-NEXT:   %nullchk = icmp eq i32* %q, null
; CHECK-NEXT:   %ret2 = call i32 @spec(i32* nonnull %q)
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %isnull ]
  %ret = call i32 @load(i32* %loc)
  %nullchk = icmp eq i32* %q, null
  br i1 %nullchk, label %isnull, label %nonnullbb

nonnullbb:  
  %ret2 = call i32 @spec(i32* nonnull %q)
  br label %isnull

isnull:  
  store volatile i32 %ret, i32* %sink
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @store(i32 %val, i32* %p) argmemonly writeonly nounwind

define void @test(i32* %loc) {
; CHECK-LABEL: @test
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, i32* %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @test_multiexit(i32* %loc, i1 %earlycnd) {
; CHECK-LABEL: @test_multiexit
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: backedge:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  call void @store(i32 0, i32* %loc)
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

define void @neg_lv_value(i32* %loc) {
; CHECK-LABEL: @neg_lv_value
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 %iv, i32* %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_lv_addr(i32* %loc) {
; CHECK-LABEL: @neg_lv_addr
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %p = getelementptr i32, i32* %loc, i32 %iv
  call void @store(i32 0, i32* %p)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_mod(i32* %loc) {
; CHECK-LABEL: @neg_mod
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, i32* %loc)
  store i32 %iv, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_ref(i32* %loc) {
; CHECK-LABEL: @neg_ref
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit1:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  call void @store(i32 0, i32* %loc)
  %v = load i32, i32* %loc
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

define void @neg_modref(i32* %loc) {
; CHECK-LABEL: @neg_modref
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, i32* %loc)
  call void @modref()
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_fence(i32* %loc) {
; CHECK-LABEL: @neg_fence
; CHECK-LABEL: loop:
; CHECK: call void @store
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @store(i32 0, i32* %loc)
  fence seq_cst
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @not_nounwind(i32 %v, i32* %p) writeonly argmemonly
declare void @not_argmemonly(i32 %v, i32* %p) writeonly nounwind
declare void @not_writeonly(i32 %v, i32* %p) argmemonly nounwind

define void @neg_not_nounwind(i32* %loc) {
; CHECK-LABEL: @neg_not_nounwind
; CHECK-LABEL: loop:
; CHECK: call void @not_nounwind
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @not_nounwind(i32 0, i32* %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_not_argmemonly(i32* %loc) {
; CHECK-LABEL: @neg_not_argmemonly
; CHECK-LABEL: loop:
; CHECK: call void @not_argmemonly
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @not_argmemonly(i32 0, i32* %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_not_writeonly(i32* %loc) {
; CHECK-LABEL: @neg_not_writeonly
; CHECK-LABEL: loop:
; CHECK: call void @not_writeonly
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @not_writeonly(i32 0, i32* %loc)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

