; REQUIRES: asserts
; RUN: opt -licm -basicaa -ipt-expensive-asserts=true < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -ipt-expensive-asserts=true < %s -S | FileCheck %s

; Hoist guard and load.
define void @test1(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test1(
; CHECK-LABEL: entry:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 %cond)
; CHECK: %val = load i32, i32* %ptr
; CHECK-LABEL: loop:

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) ["deopt" (i32 0)]
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

; Can't hoist over a side effect
define void @test2(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test2(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 %cond)
; CHECK: %val = load i32, i32* %ptr

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  store i32 0, i32* %ptr
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) ["deopt" (i32 0)]
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}

; Can't hoist over a side effect
define void @test2b(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test2b(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 %cond)
; CHECK: %val = load i32, i32* %ptr

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  %p2 = getelementptr i32, i32* %ptr, i32 1
  store i32 0, i32* %p2
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) ["deopt" (i32 0)]
  %val = load i32, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}


; Hoist guard. Cannot hoist load because of aliasing.
define void @test3(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @test3(
; CHECK-LABEL: entry:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 %cond)
; CHECK-LABEL: loop:
; CHECK: %val = load i32, i32* %ptr
; CHECK: store i32 0, i32* %ptr

entry:
  br label %loop

loop:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %loop ]
  call void (i1, ...) @llvm.experimental.guard(i1 %cond) ["deopt" (i32 0)]
  %val = load i32, i32* %ptr
  store i32 0, i32* %ptr
  %x.inc = add i32 %x, %val
  br label %loop
}


define void @test4(i1 %c, i32* %p) {

; CHECK-LABEL: @test4(
; CHECK-LABEL: entry:
; CHECK:       %a = load i32, i32* %p
; CHECK:       %invariant_cond = icmp ne i32 %a, 100
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)
; CHECK-LABEL: loop:
; CHECK-LABEL: backedge:

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  br i1 %c, label %if.true, label %if.false

if.true:
  br label %backedge

if.false:
  br label %backedge

backedge:
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Do not hoist across a conditionally executed side effect.
define void @test4a(i1 %c, i32* %p, i32* %q) {

; CHECK-LABEL: @test4a(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK-LABEL: if.true:
; CHECK:       store
; CHECK-LABEL: backedge:
; CHECK:       %a = load i32, i32* %p
; CHECK:       %invariant_cond = icmp ne i32 %a, 100
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  br i1 %c, label %if.true, label %if.false

if.true:
  store i32 123, i32* %q
  br label %backedge

if.false:
  br label %backedge

backedge:
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Do not hoist a conditionally executed guard.
define void @test4b(i1 %c, i32* %p, i32* %q) {

; CHECK-LABEL: @test4b(
; CHECK-LABEL: entry:
; CHECK-LABEL: loop:
; CHECK-LABEL: if.true:
; CHECK:       %a = load i32, i32* %p
; CHECK:       %invariant_cond = icmp ne i32 %a, 100
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)
; CHECK-LABEL: backedge:

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  br i1 %c, label %if.true, label %if.false

if.true:
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  br label %backedge

if.false:
  br label %backedge

backedge:
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Check that we don't hoist across a store in the header.
define void @test4c(i1 %c, i32* %p, i8* noalias %s) {

; CHECK-LABEL: @test4c(
; CHECK-LABEL: entry:
; CHECK:       %a = load i32, i32* %p
; CHECK:       %invariant_cond = icmp ne i32 %a, 100
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)
; CHECK-LABEL: loop:
; CHECK-LABEL: backedge:

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  store i8 0, i8* %s
  br i1 %c, label %if.true, label %if.false

if.true:
  br label %backedge

if.false:
  br label %backedge

backedge:
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Check that we don't hoist across a store in a conditionally execute block.
define void @test4d(i1 %c, i32* %p, i8* noalias %s) {

; CHECK-LABEL: @test4d(
; CHECK-LABEL: entry:
; CHECK:       %a = load i32, i32* %p
; CHECK:       %invariant_cond = icmp ne i32 %a, 100
; CHECK-LABEL: loop:
; CHECK-LABEL: backedge:
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  br i1 %c, label %if.true, label %if.false

if.true:
  store i8 0, i8* %s
  br label %backedge

if.false:
  br label %backedge

backedge:
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Check that we don't hoist across a store before the guard in the backedge.
define void @test4e(i1 %c, i32* %p, i8* noalias %s) {

; CHECK-LABEL: @test4e(
; CHECK-LABEL: entry:
; CHECK:       %a = load i32, i32* %p
; CHECK:       %invariant_cond = icmp ne i32 %a, 100
; CHECK:       store i8 0, i8* %s
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)
; CHECK-LABEL: loop:
; CHECK-LABEL: backedge:

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  br i1 %c, label %if.true, label %if.false

if.true:
  br label %backedge

if.false:
  br label %backedge

backedge:
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  store i8 0, i8* %s
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Check that we can hoist the guard in spite of store which happens after.
define void @test4f(i1 %c, i32* %p, i8* noalias %s) {

; CHECK-LABEL: @test4f(
; CHECK-LABEL: entry:
; CHECK:       %a = load i32, i32* %p
; CHECK:       %invariant_cond = icmp ne i32 %a, 100
; CHECK:       call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)
; CHECK-LABEL: loop:
; CHECK-LABEL: backedge:

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  br i1 %c, label %if.true, label %if.false

if.true:
  br label %backedge

if.false:
  br label %backedge

backedge:
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  store i8 0, i8* %s
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Do not hoist an invariant guard across a variant guard.
define void @test5(i1 %c, i32* %p, i32* %q) {

; CHECK-LABEL: @test5(
; CHECK-LABEL: entry:
; CHECK:         %a = load i32, i32* %p
; CHECK:         %invariant_cond = icmp ne i32 %a, 100
; CHECK-LABEL: loop:
; CHECK:         %variant_cond = icmp ne i32 %a, %iv
; CHECK:         call void (i1, ...) @llvm.experimental.guard(i1 %variant_cond)
; CHECK:         call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  %variant_cond = icmp ne i32 %a, %iv
  call void (i1, ...) @llvm.experimental.guard(i1 %variant_cond) [ "deopt"() ]
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  br label %backedge

backedge:
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

; Hoist an invariant guard, leave the following variant guard in the loop.
define void @test5a(i1 %c, i32* %p, i32* %q) {

; CHECK-LABEL: @test5a(
; CHECK-LABEL: entry:
; CHECK:         %a = load i32, i32* %p
; CHECK:         %invariant_cond = icmp ne i32 %a, 100
; CHECK:         call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond)
; CHECK-LABEL: loop:
; CHECK:         %variant_cond = icmp ne i32 %a, %iv
; CHECK:         call void (i1, ...) @llvm.experimental.guard(i1 %variant_cond)

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %iv.next = add i32 %iv, 1
  %a = load i32, i32* %p
  %invariant_cond = icmp ne i32 %a, 100
  %variant_cond = icmp ne i32 %a, %iv
  call void (i1, ...) @llvm.experimental.guard(i1 %invariant_cond) [ "deopt"() ]
  call void (i1, ...) @llvm.experimental.guard(i1 %variant_cond) [ "deopt"() ]
  br label %backedge

backedge:
  %loop_cond = icmp slt i32 %iv.next, 1000
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

declare void @llvm.experimental.guard(i1, ...)
