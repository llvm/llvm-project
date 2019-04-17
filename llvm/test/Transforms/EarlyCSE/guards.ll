; RUN: opt -S -early-cse < %s | FileCheck %s
; RUN: opt < %s -S -basicaa -early-cse-memssa | FileCheck %s

declare void @llvm.experimental.guard(i1,...)

declare void @llvm.assume(i1)

define i32 @test0(i32* %ptr, i1 %cond) {
; We can do store to load forwarding over a guard, since it does not
; clobber memory

; CHECK-LABEL: @test0(
; CHECK-NEXT:  store i32 40, i32* %ptr
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
; CHECK-NEXT:  ret i32 40

  store i32 40, i32* %ptr
  call void(i1,...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %rval = load i32, i32* %ptr
  ret i32 %rval
}

define i32 @test1(i32* %val, i1 %cond) {
; We can CSE loads over a guard, since it does not clobber memory

; CHECK-LABEL: @test1(
; CHECK-NEXT:  %val0 = load i32, i32* %val
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
; CHECK-NEXT:  ret i32 0

  %val0 = load i32, i32* %val
  call void(i1,...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %val1 = load i32, i32* %val
  %rval = sub i32 %val0, %val1
  ret i32 %rval
}

define i32 @test2() {
; Guards on "true" get removed

; CHECK-LABEL: @test2(
; CHECK-NEXT: ret i32 0
  call void(i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
  ret i32 0
}

define i32 @test3(i32 %val) {
; After a guard has executed the condition it was guarding is known to
; be true.

; CHECK-LABEL: @test3(
; CHECK-NEXT:  %cond0 = icmp slt i32 %val, 40
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond0) [ "deopt"() ]
; CHECK-NEXT:  ret i32 -1

  %cond0 = icmp slt i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond0) [ "deopt"() ]
  %cond1 = icmp slt i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]

  %cond2 = icmp slt i32 %val, 40
  %rval = sext i1 %cond2 to i32
  ret i32 %rval
}

define i32 @test3.unhandled(i32 %val) {
; After a guard has executed the condition it was guarding is known to
; be true.

; CHECK-LABEL: @test3.unhandled(
; CHECK-NEXT:  %cond0 = icmp slt i32 %val, 40
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond0) [ "deopt"() ]
; CHECK-NEXT:  %cond1 = icmp sge i32 %val, 40
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
; CHECK-NEXT:  ret i32 0

; Demonstrates a case we do not yet handle (it is legal to fold %cond2
; to false)
  %cond0 = icmp slt i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond0) [ "deopt"() ]
  %cond1 = icmp sge i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
  ret i32 0
}

define i32 @test4(i32 %val, i1 %c) {
; Same as test3, but with some control flow involved.

; CHECK-LABEL: @test4(
; CHECK: entry:
; CHECK-NEXT:  %cond0 = icmp slt i32 %val, 40
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond0
; CHECK-NEXT:  br label %bb0

; CHECK:     bb0:
; CHECK-NEXT:  %cond2 = icmp ult i32 %val, 200
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond2
; CHECK-NEXT:  br i1 %c, label %left, label %right

; CHECK:     left:
; CHECK-NEXT:  ret i32 0

; CHECK:     right:
; CHECK-NEXT:  ret i32 20

entry:
  %cond0 = icmp slt i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond0) [ "deopt"() ]
  %cond1 = icmp slt i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
  br label %bb0

bb0:
  %cond2 = icmp ult i32 %val, 200
  call void(i1,...) @llvm.experimental.guard(i1 %cond2) [ "deopt"() ]
  br i1 %c, label %left, label %right

left:
  %cond3 = icmp ult i32 %val, 200
  call void(i1,...) @llvm.experimental.guard(i1 %cond3) [ "deopt"() ]
  ret i32 0

right:
 ret i32 20
}

define i32 @test5(i32 %val, i1 %c) {
; Same as test4, but the %left block has mutliple predecessors.

; CHECK-LABEL: @test5(

; CHECK: entry:
; CHECK-NEXT:  %cond0 = icmp slt i32 %val, 40
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond0
; CHECK-NEXT:  br label %bb0

; CHECK: bb0:
; CHECK-NEXT:  %cond2 = icmp ult i32 %val, 200
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cond2
; CHECK-NEXT:  br i1 %c, label %left, label %right

; CHECK: left:
; CHECK-NEXT:  br label %right

; CHECK: right:
; CHECK-NEXT:  br label %left

entry:
  %cond0 = icmp slt i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond0) [ "deopt"() ]
  %cond1 = icmp slt i32 %val, 40
  call void(i1,...) @llvm.experimental.guard(i1 %cond1) [ "deopt"() ]
  br label %bb0

bb0:
  %cond2 = icmp ult i32 %val, 200
  call void(i1,...) @llvm.experimental.guard(i1 %cond2) [ "deopt"() ]
  br i1 %c, label %left, label %right

left:
  %cond3 = icmp ult i32 %val, 200
  call void(i1,...) @llvm.experimental.guard(i1 %cond3) [ "deopt"() ]
  br label %right

right:
  br label %left
}

define void @test6(i1 %c, i32* %ptr) {
; Check that we do not DSE over calls to @llvm.experimental.guard.
; Guard intrinsics do _read_ memory, so th call to guard below needs
; to see the store of 500 to %ptr

; CHECK-LABEL: @test6(
; CHECK-NEXT:  store i32 500, i32* %ptr
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %c) [ "deopt"() ]
; CHECK-NEXT:  store i32 600, i32* %ptr


  store i32 500, i32* %ptr
  call void(i1,...) @llvm.experimental.guard(i1 %c) [ "deopt"() ]
  store i32 600, i32* %ptr
  ret void
}

define void @test07(i32 %a, i32 %b) {
; Check that we are able to remove the guards on the same condition even if the
; condition is not being recalculated.

; CHECK-LABEL: @test07(
; CHECK-NEXT:  %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
; CHECK-NEXT:  ret void

  %cmp = icmp eq i32 %a, %b
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  ret void
}

define void @test08(i32 %a, i32 %b, i32* %ptr) {
; Check that we deal correctly with stores when removing guards in the same
; block in case when the condition is not recalculated.

; CHECK-LABEL: @test08(
; CHECK-NEXT:  %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:  store i32 100, i32* %ptr
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
; CHECK-NEXT:  store i32 400, i32* %ptr
; CHECK-NEXT:  ret void

  %cmp = icmp eq i32 %a, %b
  store i32 100, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 200, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 300, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 400, i32* %ptr
  ret void
}

define void @test09(i32 %a, i32 %b, i1 %c, i32* %ptr) {
; Similar to test08, but with more control flow.
; TODO: Can we get rid of the store in the end of entry given that it is
; post-dominated by other stores?

; CHECK-LABEL: @test09(
; CHECK:       entry:
; CHECK-NEXT:    %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:    store i32 100, i32* %ptr
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
; CHECK-NEXT:    store i32 400, i32* %ptr
; CHECK-NEXT:    br i1 %c, label %if.true, label %if.false
; CHECK:       if.true:
; CHECK-NEXT:    store i32 500, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       if.false:
; CHECK-NEXT:    store i32 600, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    ret void

entry:
  %cmp = icmp eq i32 %a, %b
  store i32 100, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 200, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 300, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 400, i32* %ptr
  br i1 %c, label %if.true, label %if.false

if.true:
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 500, i32* %ptr
  br label %merge

if.false:
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 600, i32* %ptr
  br label %merge

merge:
  ret void
}

define void @test10(i32 %a, i32 %b, i1 %c, i32* %ptr) {
; Make sure that non-dominating guards do not cause other guards removal.

; CHECK-LABEL: @test10(
; CHECK:       entry:
; CHECK-NEXT:    %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:    br i1 %c, label %if.true, label %if.false
; CHECK:       if.true:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
; CHECK-NEXT:    store i32 100, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       if.false:
; CHECK-NEXT:    store i32 200, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    store i32 300, i32* %ptr
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
; CHECK-NEXT:    store i32 400, i32* %ptr
; CHECK-NEXT:    ret void

entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %c, label %if.true, label %if.false

if.true:
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 100, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  br label %merge

if.false:
  store i32 200, i32* %ptr
  br label %merge

merge:
  store i32 300, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 400, i32* %ptr
  ret void
}

define void @test11(i32 %a, i32 %b, i32* %ptr) {
; Make sure that branching condition is applied to guards.

; CHECK-LABEL: @test11(
; CHECK:       entry:
; CHECK-NEXT:    %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:    br i1 %cmp, label %if.true, label %if.false
; CHECK:       if.true:
; CHECK-NEXT:    br label %merge
; CHECK:       if.false:
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 false) [ "deopt"() ]
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    ret void

entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.true, label %if.false

if.true:
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  br label %merge

if.false:
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  br label %merge

merge:
  ret void
}

define void @test12(i32 %a, i32 %b) {
; Check that the assume marks its condition as being true (and thus allows to
; eliminate the dominated guards).

; CHECK-LABEL: @test12(
; CHECK-NEXT:  %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:  call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:  ret void

  %cmp = icmp eq i32 %a, %b
  call void @llvm.assume(i1 %cmp)
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  ret void
}

define void @test13(i32 %a, i32 %b, i32* %ptr) {
; Check that we deal correctly with stores when removing guards due to assume.

; CHECK-LABEL: @test13(
; CHECK-NEXT:  %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:  call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:  store i32 400, i32* %ptr
; CHECK-NEXT:  ret void

  %cmp = icmp eq i32 %a, %b
  call void @llvm.assume(i1 %cmp)
  store i32 100, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 200, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 300, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 400, i32* %ptr
  ret void
}

define void @test14(i32 %a, i32 %b, i1 %c, i32* %ptr) {
; Similar to test13, but with more control flow.
; TODO: Can we get rid of the store in the end of entry given that it is
; post-dominated by other stores?

; CHECK-LABEL: @test14(
; CHECK:       entry:
; CHECK-NEXT:    %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:    call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:    store i32 400, i32* %ptr
; CHECK-NEXT:    br i1 %c, label %if.true, label %if.false
; CHECK:       if.true:
; CHECK-NEXT:    store i32 500, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       if.false:
; CHECK-NEXT:    store i32 600, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    ret void

entry:
  %cmp = icmp eq i32 %a, %b
  call void @llvm.assume(i1 %cmp)
  store i32 100, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 200, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 300, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 400, i32* %ptr
  br i1 %c, label %if.true, label %if.false

if.true:
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 500, i32* %ptr
  br label %merge

if.false:
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 600, i32* %ptr
  br label %merge

merge:
  ret void
}

define void @test15(i32 %a, i32 %b, i1 %c, i32* %ptr) {
; Make sure that non-dominating assumes do not cause guards removal.

; CHECK-LABEL: @test15(
; CHECK:       entry:
; CHECK-NEXT:    %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:    br i1 %c, label %if.true, label %if.false
; CHECK:       if.true:
; CHECK-NEXT:    call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:    store i32 100, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       if.false:
; CHECK-NEXT:    store i32 200, i32* %ptr
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    store i32 300, i32* %ptr
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
; CHECK-NEXT:    store i32 400, i32* %ptr
; CHECK-NEXT:    ret void

entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %c, label %if.true, label %if.false

if.true:
  call void @llvm.assume(i1 %cmp)
  store i32 100, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  br label %merge

if.false:
  store i32 200, i32* %ptr
  br label %merge

merge:
  store i32 300, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  store i32 400, i32* %ptr
  ret void
}

define void @test16(i32 %a, i32 %b) {
; Check that we don't bother to do anything with assumes even if we know the
; condition being true.

; CHECK-LABEL: @test16(
; CHECK-NEXT:    %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:    call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:    call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:    ret void

  %cmp = icmp eq i32 %a, %b
  call void @llvm.assume(i1 %cmp)
  call void @llvm.assume(i1 %cmp)
  ret void
}

define void @test17(i32 %a, i32 %b, i1 %c, i32* %ptr) {
; Check that we don't bother to do anything with assumes even if we know the
; condition being true or false (includes come control flow).

; CHECK-LABEL: @test17(
; CHECK:       entry:
; CHECK-NEXT:    %cmp = icmp eq i32 %a, %b
; CHECK-NEXT:    br i1 %c, label %if.true, label %if.false
; CHECK:       if.true:
; CHECK-NEXT:    call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:    br label %merge
; CHECK:       if.false:
; CHECK-NEXT:    call void @llvm.assume(i1 %cmp)
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    ret void

entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %c, label %if.true, label %if.false

if.true:
  call void @llvm.assume(i1 %cmp)
  br label %merge

if.false:
  call void @llvm.assume(i1 %cmp)
  br label %merge

merge:
  ret void
}

define void @test18(i1 %c) {
; Check that we don't bother to do anything with assumes even if we know the
; condition being true and not being an instruction.

; CHECK-LABEL: @test18(
; CHECK-NEXT:    call void @llvm.assume(i1 %c)
; CHECK-NEXT:    call void @llvm.assume(i1 %c)
; CHECK-NEXT:    ret void

  call void @llvm.assume(i1 %c)
  call void @llvm.assume(i1 %c)
  ret void
}
