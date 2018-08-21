; RUN: opt -S -basicaa -licm %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' < %s -S | FileCheck %s

define void @test(i32* %loc) {
; CHECK-LABEL: @test
; CHECK-LABEL: exit:
; CHECK: store i32 0, i32* %loc
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
; CHECK-LABEL: exit1:
; CHECK: store i32 0, i32* %loc
; CHECK-LABEL: exit2:
; CHECK: store i32 0, i32* %loc
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


