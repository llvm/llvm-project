; RUN: opt -S -basic-aa -licm -enable-mssa-loop-dependency=false %s -enable-new-pm=0 | FileCheck -check-prefixes=CHECK,AST %s
; RUN: opt -S -basic-aa -licm -enable-mssa-loop-dependency=true %s -enable-new-pm=0 | FileCheck  -check-prefixes=CHECK,MSSA %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck -check-prefixes=CHECK,AST %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck -check-prefixes=CHECK,MSSA %s

define void @test(i32* %loc) {
; CHECK-LABEL: @test
; CHECK-LABEL: entry:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @test_multiexit(i32* %loc, i1 %earlycnd) {
; CHECK-LABEL: @test_multiexit
; CHECK-LABEL: entry:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  store i32 0, i32* %loc
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

define i32* @false_negative_2use(i32* %loc) {
; CHECK-LABEL: @false_negative_2use
; AST-LABEL: exit:
; AST: store i32 0, i32* %loc
; MSSA-LABEL: entry:
; MSSA: store i32 0, i32* %loc
; MSSA-LABEL: loop:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32* %loc
}

define void @neg_lv_value(i32* %loc) {
; CHECK-LABEL: @neg_lv_value
; CHECK-LABEL: exit:
; CHECK: store i32 %iv.lcssa, i32* %loc
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 %iv, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_lv_addr(i32* %loc) {
; CHECK-LABEL: @neg_lv_addr
; CHECK-LABEL: loop:
; CHECK: store i32 0, i32* %p
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %p = getelementptr i32, i32* %loc, i32 %iv
  store i32 0, i32* %p
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_mod(i32* %loc) {
; CHECK-LABEL: @neg_mod
; CHECK-LABEL: exit:
; CHECK: store i32 %iv.lcssa, i32* %loc
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  store i32 %iv, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Hoisting the store is actually valid here, as it dominates the load.
define void @neg_ref(i32* %loc) {
; CHECK-LABEL: @neg_ref
; CHECK-LABEL: exit1:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit2:
; CHECK: store i32 0, i32* %loc
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  store i32 0, i32* %loc
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

; Hoisting the store here leads to a miscompile.
define void @neg_ref2(i32* %loc) {
; CHECK-LABEL: @neg_ref2
; CHECK-LABEL: exit1:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit2:
; CHECK: store i32 0, i32* %loc
entry:
  store i32 198, i32* %loc
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  %v = load i32, i32* %loc
  store i32 0, i32* %loc
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
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
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
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  fence seq_cst
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_volatile(i32* %loc) {
; CHECK-LABEL: @neg_volatile
; CHECK-LABEL: loop:
; CHECK: store volatile i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store volatile i32 0, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_release(i32* %loc) {
; CHECK-LABEL: @neg_release
; CHECK-LABEL: loop:
; CHECK: store atomic i32 0, i32* %loc release, align 4
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store atomic i32 0, i32* %loc release, align 4
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_seq_cst(i32* %loc) {
; CHECK-LABEL: @neg_seq_cst
; CHECK-LABEL: loop:
; CHECK: store atomic i32 0, i32* %loc seq_cst, align 4
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store atomic i32 0, i32* %loc seq_cst, align 4
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @maythrow() inaccessiblememonly

define void @neg_early_exit(i32* %loc) {
; CHECK-LABEL: @neg_early_exit
; CHECK-LABEL: body:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %body]
  %is_null = icmp eq i32* %loc, null
  br i1 %is_null, label %exit, label %body
body:
  call void @maythrow()
  store i32 0, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @neg_early_throw(i32* %loc) {
; CHECK-LABEL: @neg_early_throw
; CHECK-LABEL: loop:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  call void @maythrow()
  store i32 0, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @test_late_throw(i32* %loc) {
; CHECK-LABEL: @test_late_throw
; CHECK-LABEL: entry:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: loop:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  call void @maythrow()
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; TODO: could validly hoist the store here since we know what value
; the load must observe.
define i32 @test_dominated_read(i32* %loc) {
; CHECK-LABEL: @test_dominated_read
; MSSA-LABEL: entry:
; MSSA: store i32 0, i32* %loc
; MSSA-LABEL: loop:
; AST-LABEL: exit:
; AST:  store i32 0, i32* %loc
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  %reload = load i32, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %reload
}

; TODO: could validly hoist the store since we already hoisted the load and
; it's no longer in the loop.
define i32 @test_dominating_read(i32* %loc) {
; CHECK-LABEL: @test_dominating_read
; CHECK-LABEL: exit:
; CHECK: store i32 0, i32* %loc
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %reload = load i32, i32* %loc
  store i32 0, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %reload
}

declare void @readonly() readonly

; TODO: can legally hoist since value read by call is known
define void @test_dominated_readonly(i32* %loc) {
; CHECK-LABEL: @test_dominated_readonly
; CHECK-LABEL: loop:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  call void @readonly()
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; While technically possible to hoist the store to %loc, this runs across
; a funemental limitation of alias sets since both stores and the call are
; within the same alias set and we can't distinguish them cheaply.
define void @test_aliasset_fn(i32* %loc, i32* %loc2) {
; CHECK-LABEL: @test_aliasset_fn
; CHECK-LABEL: loop:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  store i32 0, i32* %loc
  call void @readonly()
  store i32 %iv, i32* %loc2
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}


; If we can't tell if the value is read before the write, we can't hoist the
; write over the potential read (since we don't know the value read)
define void @neg_may_read(i32* %loc, i1 %maybe) {
; CHECK-LABEL: @neg_may_read
; CHECK-LABEL: loop:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit:
entry:
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %merge]
  ;; maybe is a placeholder for an unanalyzable condition
  br i1 %maybe, label %taken, label %merge
taken:
  call void @readonly()
  br label %merge
merge:
  store i32 0, i32* %loc
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv, 200
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
