; REQUIRES: asserts
; RUN: opt -S -passes=licm -ipt-expensive-asserts=true < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop-mssa(licm)' -ipt-expensive-asserts=true -S %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f() nounwind
declare void @llvm.experimental.guard(i1,...)

; constant fold on first ieration
define i32 @test1(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test1(
entry:
; CHECK: %i1 = load i32, ptr %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %r.chk = icmp ult i32 %iv, 2000
  br i1 %r.chk, label %continue, label %fail
continue:
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; Same as test1, but with a floating point IR and fcmp
define i32 @test_fcmp(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test_fcmp(
entry:
; CHECK: %i1 = load i32, ptr %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi float [ 0.0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %r.chk = fcmp olt float %iv, 2000.0
  br i1 %r.chk, label %continue, label %fail
continue:
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = fadd float %iv, 1.0
  %exitcond = fcmp ogt float %inc, 1000.0
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; Count down from a.length w/entry guard
; TODO: currently unable to prove the following:
; ule i32 (add nsw i32 %len, -1), %len where len is [0, 512]
define i32 @test2(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test2(
entry:
  %len = load i32, ptr %a, align 4, !range !{i32 0, i32 512}
  %is.non.pos = icmp eq i32 %len, 0
  br i1 %is.non.pos, label %fail, label %preheader
preheader:
  %lenminusone = add nsw i32 %len, -1
  br label %for.body
for.body:
  %iv = phi i32 [ %lenminusone, %preheader ], [ %dec, %continue ]
  %acc = phi i32 [ 0, %preheader ], [ %add, %continue ]
  %r.chk = icmp ule i32 %iv, %len
  br i1 %r.chk, label %continue, label %fail
continue:
; CHECK-LABEL: continue
; CHECK: %i1 = load i32, ptr %a, align 4
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %dec = add nsw i32 %iv, -1
  %exitcond = icmp eq i32 %dec, 0
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; trivially true for zero
define i32 @test3(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test3(
entry:
  %len = load i32, ptr %a, align 4, !range !{i32 0, i32 512}
  %is.zero = icmp eq i32 %len, 0
  br i1 %is.zero, label %fail, label %preheader
preheader:
; CHECK: %i1 = load i32, ptr %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body
for.body:
  %iv = phi i32 [ 0, %preheader ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %preheader ], [ %add, %continue ]
  %r.chk = icmp ule i32 %iv, %len
  br i1 %r.chk, label %continue, label %fail
continue:
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; requires fact length is non-zero
define i32 @test4(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[LEN:%.*]] = load i32, ptr [[A:%.*]], align 4, !range !0
; CHECK-NEXT:    [[IS_ZERO:%.*]] = icmp eq i32 [[LEN]], 0
; CHECK-NEXT:    br i1 [[IS_ZERO]], label [[FAIL:%.*]], label [[PREHEADER:%.*]]
; CHECK:       preheader:
; CHECK-NEXT:    [[I1:%.*]] = load i32, ptr [[A]], align 4
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, [[PREHEADER]] ], [ [[INC:%.*]], [[CONTINUE:%.*]] ]
; CHECK-NEXT:    [[ACC:%.*]] = phi i32 [ 0, [[PREHEADER]] ], [ [[ADD:%.*]], [[CONTINUE]] ]
; CHECK-NEXT:    [[R_CHK:%.*]] = icmp ult i32 [[IV]], [[LEN]]
; CHECK-NEXT:    br i1 [[R_CHK]], label [[CONTINUE]], label [[FAIL_LOOPEXIT:%.*]]
; CHECK:       continue:
; CHECK-NEXT:    [[ADD]] = add nsw i32 [[I1]], [[ACC]]
; CHECK-NEXT:    [[INC]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[INC]], 1000
; CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_COND_CLEANUP:%.*]], label [[FOR_BODY]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    [[ADD_LCSSA:%.*]] = phi i32 [ [[ADD]], [[CONTINUE]] ]
; CHECK-NEXT:    ret i32 [[ADD_LCSSA]]
; CHECK:       fail.loopexit:
; CHECK-NEXT:    br label [[FAIL]]
; CHECK:       fail:
; CHECK-NEXT:    call void @f()
; CHECK-NEXT:    ret i32 -1
;
entry:
  %len = load i32, ptr %a, align 4, !range !{i32 0, i32 512}
  %is.zero = icmp eq i32 %len, 0
  br i1 %is.zero, label %fail, label %preheader
preheader:
  br label %for.body
for.body:
  %iv = phi i32 [ 0, %preheader ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %preheader ], [ %add, %continue ]
  %r.chk = icmp ult i32 %iv, %len
  br i1 %r.chk, label %continue, label %fail
continue:
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; variation on test1 with branch swapped
define i32 @test-brswap(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-brswap(
entry:
; CHECK: %i1 = load i32, ptr %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %r.chk = icmp ugt i32 %iv, 2000
  br i1 %r.chk, label %fail, label %continue
continue:
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

define i32 @test-nonphi(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-nonphi(
entry:
  br label %for.body

for.body:
; CHECK-LABEL: continue
; CHECK: %i1 = load i32, ptr %a, align 4
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %xor = xor i32 %iv, 72
  %r.chk = icmp ugt i32 %xor, 2000
  br i1 %r.chk, label %fail, label %continue
continue:
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

define i32 @test-wrongphi(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-wrongphi(
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue ]
  %cond = icmp ult i32 %iv, 500
  br i1 %cond, label %dummy_block1, label %dummy_block2

dummy_block1:
  br label %dummy_block2

dummy_block2:
  %wrongphi = phi i32 [11, %for.body], [12, %dummy_block1]
  %r.chk = icmp ugt i32 %wrongphi, 2000
  br i1 %r.chk, label %fail, label %continue
continue:
; CHECK-LABEL: continue
; CHECK: %i1 = load i32, ptr %a, align 4
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

; This works because loop-simplify is run implicitly, but test for it anyways
define i32 @test-multiple-latch(ptr noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test-multiple-latch(
entry:
; CHECK: %i1 = load i32, ptr %a, align 4
; CHECK-NEXT: br label %for.body
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %continue1 ], [ %inc, %continue2 ]
  %acc = phi i32 [ 0, %entry ], [ %add, %continue1 ], [ %add, %continue2 ]
  %r.chk = icmp ult i32 %iv, 2000
  br i1 %r.chk, label %continue1, label %fail
continue1:
  %i1 = load i32, ptr %a, align 4
  %add = add nsw i32 %i1, %acc
  %inc = add nuw nsw i32 %iv, 1
  %cmp = icmp eq i32 %add, 0
  br i1 %cmp, label %continue2, label %for.body
continue2:
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add

fail:
  call void @f()
  ret i32 -1
}

define void @test-hoisting-in-presence-of-guards(i1 %c, ptr %p) {

; CHECK-LABEL: @test-hoisting-in-presence-of-guards
; CHECK:       entry:
; CHECK:         %a = load i32, ptr %p
; CHECK:         %invariant_cond = icmp ne i32 %a, 100
; CHECK:       loop:

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  %a = load i32, ptr %p
  %invariant_cond = icmp ne i32 %a, 100
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}


declare void @may_throw() inaccessiblememonly

; Test that we can sink a mustexecute load from loop header even in presence of
; throwing instructions after it.
define void @test_hoist_from_header_01(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_header_01(
; CHECK:       entry:
; CHECK-NEXT:  %load = load i32, ptr %p
; CHECK-NOT:   load i32

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  %load = load i32, ptr %p
  call void @may_throw()
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  %iv.next = add i32 %iv, %merge
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

define void @test_hoist_from_header_02(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_header_02(
; CHECK:       entry:
; CHECK-NEXT:  %load = load i32, ptr %p
; CHECK-NOT:   load i32

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  %load = load i32, ptr %p
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  call void @may_throw()
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  %iv.next = add i32 %iv, %merge
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

define void @test_hoist_from_header_03(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_header_03(
; CHECK:       entry:
; CHECK-NEXT:  %load = load i32, ptr %p
; CHECK-NOT:   load i32

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  %load = load i32, ptr %p
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  call void @may_throw()
  %iv.next = add i32 %iv, %merge
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

; Check that a throwing instruction prohibits hoisting across it.
define void @test_hoist_from_header_04(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_header_04(
; CHECK:       entry:
; CHECK:       loop:
; CHECK:       %load = load i32, ptr %p

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  call void @may_throw()
  %load = load i32, ptr %p
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  %iv.next = add i32 %iv, %merge
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

; Check that we can hoist a mustexecute load from backedge even if something
; throws after it.
define void @test_hoist_from_backedge_01(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_backedge_01(
; CHECK:       entry:
; CHECK-NEXT:  %load = load i32, ptr %p
; CHECK-NOT:   load i32

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  %iv.next = add i32 %iv, %merge
  %load = load i32, ptr %p
  call void @may_throw()
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

; Check that we don't hoist the load if something before it can throw.
define void @test_hoist_from_backedge_02(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_backedge_02(
; CHECK:       entry:
; CHECK:       loop:
; CHECK:       %load = load i32, ptr %p

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  %iv.next = add i32 %iv, %merge
  call void @may_throw()
  %load = load i32, ptr %p
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

define void @test_hoist_from_backedge_03(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_backedge_03(
; CHECK:       entry:
; CHECK:       loop:
; CHECK:       %load = load i32, ptr %p

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  call void @may_throw()
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  %iv.next = add i32 %iv, %merge
  %load = load i32, ptr %p
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}

define void @test_hoist_from_backedge_04(ptr %p, i32 %n) {

; CHECK-LABEL: @test_hoist_from_backedge_04(
; CHECK:       entry:
; CHECK:       loop:
; CHECK:       %load = load i32, ptr %p

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %dummy = phi i32 [ 0, %entry ], [ %merge, %backedge ]
  call void @may_throw()
  %cond = icmp slt i32 %iv, %n
  br i1 %cond, label %if.true, label %if.false

if.true:
  %a = add i32 %iv, %iv
  br label %backedge

if.false:
  %b = mul i32 %iv, %iv
  br label %backedge

backedge:
  %merge = phi i32 [ %a, %if.true ], [ %b, %if.false ]
  %iv.next = add i32 %iv, %merge
  %load = load i32, ptr %p
  %loop.cond = icmp ult i32 %iv.next, %load
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}
