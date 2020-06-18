; RUN: opt -passes='loop(unswitch),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -verify-memoryssa -passes='loop-mssa(unswitch),verify<loops>' -S < %s | FileCheck %s

declare void @some_func() noreturn
declare void @sink(i32)

declare i1 @cond()
declare i32 @cond.i32()

; This test contains two trivial unswitch condition in one loop.
; LoopUnswitch pass should be able to unswitch the second one
; after unswitching the first one.
define i32 @test1(i32* %var, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test1(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %{{.*}}, label %entry.split, label %loop_exit.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br i1 %{{.*}}, label %entry.split.split, label %loop_exit
;
; CHECK:       entry.split.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond1, label %continue, label %loop_exit	; first trivial condition
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %continue

continue:
  %var_val = load i32, i32* %var
  br i1 %cond2, label %do_something, label %loop_exit	; second trivial condition
; CHECK:       continue:
; CHECK-NEXT:    load
; CHECK-NEXT:    br label %do_something

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       do_something:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  ret i32 0
; CHECK:       loop_exit:
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    ret
}

; Test for two trivially unswitchable switches.
define i32 @test3(i32* %var, i32 %cond1, i32 %cond2) {
; CHECK-LABEL: @test3(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond1, label %entry.split [
; CHECK-NEXT:      i32 0, label %loop_exit1
; CHECK-NEXT:    ]
;
; CHECK:       entry.split:
; CHECK-NEXT:    switch i32 %cond2, label %loop_exit2 [
; CHECK-NEXT:      i32 42, label %loop_exit2
; CHECK-NEXT:      i32 0, label %entry.split.split
; CHECK-NEXT:    ]
;
; CHECK:       entry.split.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  switch i32 %cond1, label %continue [
    i32 0, label %loop_exit1
  ]
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %continue

continue:
  %var_val = load i32, i32* %var
  switch i32 %cond2, label %loop_exit2 [
    i32 0, label %do_something
    i32 42, label %loop_exit2
  ]
; CHECK:       continue:
; CHECK-NEXT:    load
; CHECK-NEXT:    br label %do_something

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       do_something:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit1:
  ret i32 0
; CHECK:       loop_exit1:
; CHECK-NEXT:    ret

loop_exit2:
  ret i32 0
; CHECK:       loop_exit2:
; CHECK-NEXT:    ret
;
; We shouldn't have any unreachable blocks here because the unswitched switches
; turn into branches instead.
; CHECK-NOT:     unreachable
}

; Test for a trivially unswitchable switch with multiple exiting cases and
; multiple looping cases.
define i32 @test4(i32* %var, i32 %cond1, i32 %cond2) {
; CHECK-LABEL: @test4(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond2, label %loop_exit2 [
; CHECK-NEXT:      i32 13, label %loop_exit1
; CHECK-NEXT:      i32 42, label %loop_exit3
; CHECK-NEXT:      i32 0, label %entry.split
; CHECK-NEXT:      i32 1, label %entry.split
; CHECK-NEXT:      i32 2, label %entry.split
; CHECK-NEXT:    ]
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %var_val = load i32, i32* %var
  switch i32 %cond2, label %loop_exit2 [
    i32 0, label %loop0
    i32 1, label %loop1
    i32 13, label %loop_exit1
    i32 2, label %loop2
    i32 42, label %loop_exit3
  ]
; CHECK:       loop_begin:
; CHECK-NEXT:    load
; CHECK-NEXT:    switch i32 %cond2, label %loop2 [
; CHECK-NEXT:      i32 0, label %loop0
; CHECK-NEXT:      i32 1, label %loop1
; CHECK-NEXT:    ]

loop0:
  call void @some_func() noreturn nounwind
  br label %loop_latch
; CHECK:       loop0:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_latch

loop1:
  call void @some_func() noreturn nounwind
  br label %loop_latch
; CHECK:       loop1:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_latch

loop2:
  call void @some_func() noreturn nounwind
  br label %loop_latch
; CHECK:       loop2:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_latch

loop_latch:
  br label %loop_begin
; CHECK:       loop_latch:
; CHECK-NEXT:    br label %loop_begin

loop_exit1:
  ret i32 0
; CHECK:       loop_exit1:
; CHECK-NEXT:    ret

loop_exit2:
  ret i32 0
; CHECK:       loop_exit2:
; CHECK-NEXT:    ret

loop_exit3:
  ret i32 0
; CHECK:       loop_exit3:
; CHECK-NEXT:    ret
}

; This test contains a trivially unswitchable branch with an LCSSA phi node in
; a loop exit block.
define i32 @test5(i1 %cond1, i32 %x, i32 %y) {
; CHECK-LABEL: @test5(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %{{.*}}, label %entry.split, label %loop_exit
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond1, label %latch, label %loop_exit
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %latch

latch:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       latch:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %result1 = phi i32 [ %x, %loop_begin ]
  %result2 = phi i32 [ %y, %loop_begin ]
  %result = add i32 %result1, %result2
  ret i32 %result
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[R1:.*]] = phi i32 [ %x, %entry ]
; CHECK-NEXT:    %[[R2:.*]] = phi i32 [ %y, %entry ]
; CHECK-NEXT:    %[[R:.*]] = add i32 %[[R1]], %[[R2]]
; CHECK-NEXT:    ret i32 %[[R]]
}

; This test contains a trivially unswitchable branch with a real phi node in LCSSA
; position in a shared exit block where a different path through the loop
; produces a non-invariant input to the PHI node.
define i32 @test6(i32* %var, i1 %cond1, i1 %cond2, i32 %x, i32 %y) {
; CHECK-LABEL: @test6(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %{{.*}}, label %entry.split, label %loop_exit.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond1, label %continue, label %loop_exit
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %continue

continue:
  %var_val = load i32, i32* %var
  br i1 %cond2, label %latch, label %loop_exit
; CHECK:       continue:
; CHECK-NEXT:    load
; CHECK-NEXT:    br i1 %cond2, label %latch, label %loop_exit

latch:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       latch:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %result1 = phi i32 [ %x, %loop_begin ], [ %var_val, %continue ]
  %result2 = phi i32 [ %var_val, %continue ], [ %y, %loop_begin ]
  %result = add i32 %result1, %result2
  ret i32 %result
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[R1:.*]] = phi i32 [ %var_val, %continue ]
; CHECK-NEXT:    %[[R2:.*]] = phi i32 [ %var_val, %continue ]
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[R1S:.*]] = phi i32 [ %x, %entry ], [ %[[R1]], %loop_exit ]
; CHECK-NEXT:    %[[R2S:.*]] = phi i32 [ %y, %entry ], [ %[[R2]], %loop_exit ]
; CHECK-NEXT:    %[[R:.*]] = add i32 %[[R1S]], %[[R2S]]
; CHECK-NEXT:    ret i32 %[[R]]
}

; This test contains a trivially unswitchable switch with an LCSSA phi node in
; a loop exit block.
define i32 @test7(i32 %cond1, i32 %x, i32 %y) {
; CHECK-LABEL: @test7(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond1, label %entry.split [
; CHECK-NEXT:      i32 0, label %loop_exit
; CHECK-NEXT:      i32 1, label %loop_exit
; CHECK-NEXT:    ]
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  switch i32 %cond1, label %latch [
    i32 0, label %loop_exit
    i32 1, label %loop_exit
  ]
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %latch

latch:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       latch:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %result1 = phi i32 [ %x, %loop_begin ], [ %x, %loop_begin ]
  %result2 = phi i32 [ %y, %loop_begin ], [ %y, %loop_begin ]
  %result = add i32 %result1, %result2
  ret i32 %result
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[R1:.*]] = phi i32 [ %x, %entry ], [ %x, %entry ]
; CHECK-NEXT:    %[[R2:.*]] = phi i32 [ %y, %entry ], [ %y, %entry ]
; CHECK-NEXT:    %[[R:.*]] = add i32 %[[R1]], %[[R2]]
; CHECK-NEXT:    ret i32 %[[R]]
}

; This test contains a trivially unswitchable switch with a real phi node in
; LCSSA position in a shared exit block where a different path through the loop
; produces a non-invariant input to the PHI node.
define i32 @test8(i32* %var, i32 %cond1, i32 %cond2, i32 %x, i32 %y) {
; CHECK-LABEL: @test8(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond1, label %entry.split [
; CHECK-NEXT:      i32 0, label %loop_exit.split
; CHECK-NEXT:      i32 1, label %loop_exit2
; CHECK-NEXT:      i32 2, label %loop_exit.split
; CHECK-NEXT:    ]
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  switch i32 %cond1, label %continue [
    i32 0, label %loop_exit
    i32 1, label %loop_exit2
    i32 2, label %loop_exit
  ]
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %continue

continue:
  %var_val = load i32, i32* %var
  switch i32 %cond2, label %latch [
    i32 0, label %loop_exit
  ]
; CHECK:       continue:
; CHECK-NEXT:    load
; CHECK-NEXT:    switch i32 %cond2, label %latch [
; CHECK-NEXT:      i32 0, label %loop_exit
; CHECK-NEXT:    ]

latch:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       latch:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %result1.1 = phi i32 [ %x, %loop_begin ], [ %x, %loop_begin ], [ %var_val, %continue ]
  %result1.2 = phi i32 [ %var_val, %continue ], [ %y, %loop_begin ], [ %y, %loop_begin ]
  %result1 = add i32 %result1.1, %result1.2
  ret i32 %result1
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[R1:.*]] = phi i32 [ %var_val, %continue ]
; CHECK-NEXT:    %[[R2:.*]] = phi i32 [ %var_val, %continue ]
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[R1S:.*]] = phi i32 [ %x, %entry ], [ %x, %entry ], [ %[[R1]], %loop_exit ]
; CHECK-NEXT:    %[[R2S:.*]] = phi i32 [ %y, %entry ], [ %y, %entry ], [ %[[R2]], %loop_exit ]
; CHECK-NEXT:    %[[R:.*]] = add i32 %[[R1S]], %[[R2S]]
; CHECK-NEXT:    ret i32 %[[R]]

loop_exit2:
  %result2.1 = phi i32 [ %x, %loop_begin ]
  %result2.2 = phi i32 [ %y, %loop_begin ]
  %result2 = add i32 %result2.1, %result2.2
  ret i32 %result2
; CHECK:       loop_exit2:
; CHECK-NEXT:    %[[R1:.*]] = phi i32 [ %x, %entry ]
; CHECK-NEXT:    %[[R2:.*]] = phi i32 [ %y, %entry ]
; CHECK-NEXT:    %[[R:.*]] = add i32 %[[R1]], %[[R2]]
; CHECK-NEXT:    ret i32 %[[R]]
}

; This test, extracted from the LLVM test suite, has an interesting dominator
; tree to update as there are edges to sibling domtree nodes within child
; domtree nodes of the unswitched node.
define void @xgets(i1 %cond1, i1* %cond2.ptr) {
; CHECK-LABEL: @xgets(
entry:
  br label %for.cond.preheader
; CHECK:       entry:
; CHECK-NEXT:    br label %for.cond.preheader

for.cond.preheader:
  br label %for.cond
; CHECK:       for.cond.preheader:
; CHECK-NEXT:    br i1 %cond1, label %for.cond.preheader.split, label %if.end17.thread.loopexit
;
; CHECK:       for.cond.preheader.split:
; CHECK-NEXT:    br label %for.cond

for.cond:
  br i1 %cond1, label %land.lhs.true, label %if.end17.thread.loopexit
; CHECK:       for.cond:
; CHECK-NEXT:    br label %land.lhs.true

land.lhs.true:
  br label %if.then20
; CHECK:       land.lhs.true:
; CHECK-NEXT:    br label %if.then20

if.then20:
  %cond2 = load volatile i1, i1* %cond2.ptr
  br i1 %cond2, label %if.then23, label %if.else
; CHECK:       if.then20:
; CHECK-NEXT:    %[[COND2:.*]] = load volatile i1, i1* %cond2.ptr
; CHECK-NEXT:    br i1 %[[COND2]], label %if.then23, label %if.else

if.else:
  br label %for.cond
; CHECK:       if.else:
; CHECK-NEXT:    br label %for.cond

if.end17.thread.loopexit:
  br label %if.end17.thread
; CHECK:       if.end17.thread.loopexit:
; CHECK-NEXT:    br label %if.end17.thread

if.end17.thread:
  br label %cleanup
; CHECK:       if.end17.thread:
; CHECK-NEXT:    br label %cleanup

if.then23:
  br label %cleanup
; CHECK:       if.then23:
; CHECK-NEXT:    br label %cleanup

cleanup:
  ret void
; CHECK:       cleanup:
; CHECK-NEXT:    ret void
}

define i32 @test_partial_condition_unswitch_and(i32* %var, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test_partial_condition_unswitch_and(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %entry.split, label %loop_exit.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br i1 %cond2, label %entry.split.split, label %loop_exit
;
; CHECK:       entry.split.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond1, label %continue, label %loop_exit
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %continue

continue:
  %var_val = load i32, i32* %var
  %var_cond = trunc i32 %var_val to i1
  %cond_and = and i1 %var_cond, %cond2
  br i1 %cond_and, label %do_something, label %loop_exit
; CHECK:       continue:
; CHECK-NEXT:    %[[VAR:.*]] = load i32
; CHECK-NEXT:    %[[VAR_COND:.*]] = trunc i32 %[[VAR]] to i1
; CHECK-NEXT:    %[[COND_AND:.*]] = and i1 %[[VAR_COND]], true
; CHECK-NEXT:    br i1 %[[COND_AND]], label %do_something, label %loop_exit

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       do_something:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  ret i32 0
; CHECK:       loop_exit:
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    ret
}

define i32 @test_partial_condition_unswitch_or(i32* %var, i1 %cond1, i1 %cond2, i1 %cond3, i1 %cond4, i1 %cond5, i1 %cond6) {
; CHECK-LABEL: @test_partial_condition_unswitch_or(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[INV_OR1:.*]] = or i1 %cond4, %cond2
; CHECK-NEXT:    %[[INV_OR2:.*]] = or i1 %[[INV_OR1]], %cond3
; CHECK-NEXT:    %[[INV_OR3:.*]] = or i1 %[[INV_OR2]], %cond1
; CHECK-NEXT:    br i1 %[[INV_OR3]], label %loop_exit.split, label %entry.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %var_val = load i32, i32* %var
  %var_cond = trunc i32 %var_val to i1
  %cond_or1 = or i1 %var_cond, %cond1
  %cond_or2 = or i1 %cond2, %cond3
  %cond_or3 = or i1 %cond_or1, %cond_or2
  %cond_xor1 = xor i1 %cond5, %var_cond
  %cond_and1 = and i1 %cond6, %var_cond
  %cond_or4 = or i1 %cond_xor1, %cond_and1
  %cond_or5 = or i1 %cond_or3, %cond_or4
  %cond_or6 = or i1 %cond_or5, %cond4
  br i1 %cond_or6, label %loop_exit, label %do_something
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[VAR:.*]] = load i32
; CHECK-NEXT:    %[[VAR_COND:.*]] = trunc i32 %[[VAR]] to i1
; CHECK-NEXT:    %[[COND_OR1:.*]] = or i1 %[[VAR_COND]], false
; CHECK-NEXT:    %[[COND_OR2:.*]] = or i1 false, false
; CHECK-NEXT:    %[[COND_OR3:.*]] = or i1 %[[COND_OR1]], %[[COND_OR2]]
; CHECK-NEXT:    %[[COND_XOR:.*]] = xor i1 %cond5, %[[VAR_COND]]
; CHECK-NEXT:    %[[COND_AND:.*]] = and i1 %cond6, %[[VAR_COND]]
; CHECK-NEXT:    %[[COND_OR4:.*]] = or i1 %[[COND_XOR]], %[[COND_AND]]
; CHECK-NEXT:    %[[COND_OR5:.*]] = or i1 %[[COND_OR3]], %[[COND_OR4]]
; CHECK-NEXT:    %[[COND_OR6:.*]] = or i1 %[[COND_OR5]], false
; CHECK-NEXT:    br i1 %[[COND_OR6]], label %loop_exit, label %do_something

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       do_something:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  ret i32 0
; CHECK:       loop_exit.split:
; CHECK-NEXT:    ret
}

define i32 @test_partial_condition_unswitch_with_lcssa_phi1(i32* %var, i1 %cond, i32 %x) {
; CHECK-LABEL: @test_partial_condition_unswitch_with_lcssa_phi1(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond, label %entry.split, label %loop_exit.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %var_val = load i32, i32* %var
  %var_cond = trunc i32 %var_val to i1
  %cond_and = and i1 %var_cond, %cond
  br i1 %cond_and, label %do_something, label %loop_exit
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[VAR:.*]] = load i32
; CHECK-NEXT:    %[[VAR_COND:.*]] = trunc i32 %[[VAR]] to i1
; CHECK-NEXT:    %[[COND_AND:.*]] = and i1 %[[VAR_COND]], true
; CHECK-NEXT:    br i1 %[[COND_AND]], label %do_something, label %loop_exit

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       do_something:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  %x.lcssa = phi i32 [ %x, %loop_begin ]
  ret i32 %x.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[LCSSA:.*]] = phi i32 [ %x, %loop_begin ]
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[LCSSA_SPLIT:.*]] = phi i32 [ %x, %entry ], [ %[[LCSSA]], %loop_exit ]
; CHECK-NEXT:    ret i32 %[[LCSSA_SPLIT]]
}

define i32 @test_partial_condition_unswitch_with_lcssa_phi2(i32* %var, i1 %cond, i32 %x, i32 %y) {
; CHECK-LABEL: @test_partial_condition_unswitch_with_lcssa_phi2(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond, label %entry.split, label %loop_exit.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %var_val = load i32, i32* %var
  %var_cond = trunc i32 %var_val to i1
  %cond_and = and i1 %var_cond, %cond
  br i1 %cond_and, label %do_something, label %loop_exit
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[VAR:.*]] = load i32
; CHECK-NEXT:    %[[VAR_COND:.*]] = trunc i32 %[[VAR]] to i1
; CHECK-NEXT:    %[[COND_AND:.*]] = and i1 %[[VAR_COND]], true
; CHECK-NEXT:    br i1 %[[COND_AND]], label %do_something, label %loop_exit

do_something:
  call void @some_func() noreturn nounwind
  br i1 %var_cond, label %loop_begin, label %loop_exit
; CHECK:       do_something:
; CHECK-NEXT:    call
; CHECK-NEXT:    br i1 %[[VAR_COND]], label %loop_begin, label %loop_exit

loop_exit:
  %xy.lcssa = phi i32 [ %x, %loop_begin ], [ %y, %do_something ]
  ret i32 %xy.lcssa
; CHECK:       loop_exit:
; CHECK-NEXT:    %[[LCSSA:.*]] = phi i32 [ %x, %loop_begin ], [ %y, %do_something ]
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    %[[LCSSA_SPLIT:.*]] = phi i32 [ %x, %entry ], [ %[[LCSSA]], %loop_exit ]
; CHECK-NEXT:    ret i32 %[[LCSSA_SPLIT]]
}

; Unswitch will not actually change the loop nest from:
;   A < B < C
define void @hoist_inner_loop0() {
; CHECK-LABEL: define void @hoist_inner_loop0(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    br label %b.header

b.header:
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[B_LATCH_SPLIT:.*]], label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    br label %c.header

c.header:
  br i1 %v1, label %b.latch, label %c.latch
; CHECK:       c.header:
; CHECK-NEXT:    br label %c.latch

c.latch:
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %b.latch
; CHECK:       c.latch:
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %b.latch

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.latch
; CHECK:       b.latch:
; CHECK-NEXT:    br label %[[B_LATCH_SPLIT]]
;
; CHECK:       [[B_LATCH_SPLIT]]:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Unswitch will transform the loop nest from:
;   A < B < C
; into
;   A < (B, C)
define void @hoist_inner_loop1(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop1(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %b.latch, label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  br i1 %v1, label %b.latch, label %c.latch
; CHECK:       c.header:
; CHECK-NEXT:    br label %c.latch

c.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %a.exit.c
; CHECK:       c.latch:
; CHECK-NEXT:    store i32 %x.a, i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %a.exit.c

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.exit.b
; CHECK:       b.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.exit.b

a.exit.c:
  br label %a.latch
; CHECK:       a.exit.c
; CHECK-NEXT:    br label %a.latch

a.exit.b:
  br label %a.latch
; CHECK:       a.exit.b:
; CHECK-NEXT:    br label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Unswitch will transform the loop nest from:
;   A < B < C
; into
;   (A < B), C
define void @hoist_inner_loop2(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop2(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %b.latch, label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_A_LCSSA:.*]] = phi i32 [ %x.a, %b.header ]
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  br i1 %v1, label %b.latch, label %c.latch
; CHECK:       c.header:
; CHECK-NEXT:    br label %c.latch

c.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %exit
; CHECK:       c.latch:
; CHECK-NEXT:    store i32 %[[X_A_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %exit

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.latch
; CHECK:       b.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Same as @hoist_inner_loop2 but with a nested loop inside the hoisted loop.
; Unswitch will transform the loop nest from:
;   A < B < C < D
; into
;   (A < B), (C < D)
define void @hoist_inner_loop3(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop3(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %b.latch, label %[[B_HEADER_SPLIT:.*]]
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_A_LCSSA:.*]] = phi i32 [ %x.a, %b.header ]
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  br i1 %v1, label %b.latch, label %c.body
; CHECK:       c.header:
; CHECK-NEXT:    br label %c.body

c.body:
  %x.c = load i32, i32* %ptr
  br label %d.header
; CHECK:       c.body:
; CHECK-NEXT:    %x.c = load i32, i32* %ptr
; CHECK-NEXT:    br label %d.header

d.header:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  store i32 %x.c, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %d.header, label %c.latch
; CHECK:       d.header:
; CHECK-NEXT:    store i32 %[[X_A_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %x.c, i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %d.header, label %c.latch

c.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %c.header, label %exit
; CHECK:       c.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %c.header, label %exit

b.latch:
  %v4 = call i1 @cond()
  br i1 %v4, label %b.header, label %a.latch
; CHECK:       b.latch:
; CHECK-NEXT:    %v4 = call i1 @cond()
; CHECK-NEXT:    br i1 %v4, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; This test is designed to exercise checking multiple remaining exits from the
; loop being unswitched.
; Unswitch will transform the loop nest from:
;   A < B < C < D
; into
;   A < B < (C, D)
define void @hoist_inner_loop4() {
; CHECK-LABEL: define void @hoist_inner_loop4(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    br label %b.header

b.header:
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    br label %c.header

c.header:
  %v1 = call i1 @cond()
  br label %d.header
; CHECK:       c.header:
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %[[C_HEADER_SPLIT:.*]], label %c.latch
;
; CHECK:       [[C_HEADER_SPLIT]]:
; CHECK-NEXT:    br label %d.header

d.header:
  br i1 %v1, label %d.exiting1, label %c.latch
; CHECK:       d.header:
; CHECK-NEXT:    br label %d.exiting1

d.exiting1:
  %v2 = call i1 @cond()
  br i1 %v2, label %d.exiting2, label %a.latch
; CHECK:       d.exiting1:
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %d.exiting2, label %a.latch

d.exiting2:
  %v3 = call i1 @cond()
  br i1 %v3, label %d.exiting3, label %loopexit.d
; CHECK:       d.exiting2:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %d.exiting3, label %loopexit.d

d.exiting3:
  %v4 = call i1 @cond()
  br i1 %v4, label %d.latch, label %b.latch
; CHECK:       d.exiting3:
; CHECK-NEXT:    %v4 = call i1 @cond()
; CHECK-NEXT:    br i1 %v4, label %d.latch, label %b.latch

d.latch:
  br label %d.header
; CHECK:       d.latch:
; CHECK-NEXT:    br label %d.header

c.latch:
  %v5 = call i1 @cond()
  br i1 %v5, label %c.header, label %loopexit.c
; CHECK:       c.latch:
; CHECK-NEXT:    %v5 = call i1 @cond()
; CHECK-NEXT:    br i1 %v5, label %c.header, label %loopexit.c

b.latch:
  br label %b.header
; CHECK:       b.latch:
; CHECK-NEXT:    br label %b.header

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

loopexit.d:
  br label %exit
; CHECK:       loopexit.d:
; CHECK-NEXT:    br label %exit

loopexit.c:
  br label %exit
; CHECK:       loopexit.c:
; CHECK-NEXT:    br label %exit

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Unswitch will transform the loop nest from:
;   A < B < C < D
; into
;   A < ((B < C), D)
define void @hoist_inner_loop5(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop5(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    br label %c.header

c.header:
  %x.c = load i32, i32* %ptr
  %v1 = call i1 @cond()
  br label %d.header
; CHECK:       c.header:
; CHECK-NEXT:    %x.c = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i1 @cond()
; CHECK-NEXT:    br i1 %v1, label %c.latch, label %[[C_HEADER_SPLIT:.*]]
;
; CHECK:       [[C_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %c.header ]
; CHECK-NEXT:    %[[X_C_LCSSA:.*]] = phi i32 [ %x.c, %c.header ]
; CHECK-NEXT:    br label %d.header

d.header:
  br i1 %v1, label %c.latch, label %d.latch
; CHECK:       d.header:
; CHECK-NEXT:    br label %d.latch

d.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  store i32 %x.c, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %d.header, label %a.latch
; CHECK:       d.latch:
; CHECK-NEXT:    store i32 %x.a, i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_C_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %d.header, label %a.latch

c.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %c.header, label %b.latch
; CHECK:       c.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %c.header, label %b.latch

b.latch:
  br label %b.header
; CHECK:       b.latch:
; CHECK-NEXT:    br label %b.header

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

; Same as `@hoist_inner_loop2` but using a switch.
; Unswitch will transform the loop nest from:
;   A < B < C
; into
;   (A < B), C
define void @hoist_inner_loop_switch(i32* %ptr) {
; CHECK-LABEL: define void @hoist_inner_loop_switch(
entry:
  br label %a.header
; CHECK:       entry:
; CHECK-NEXT:    br label %a.header

a.header:
  %x.a = load i32, i32* %ptr
  br label %b.header
; CHECK:       a.header:
; CHECK-NEXT:    %x.a = load i32, i32* %ptr
; CHECK-NEXT:    br label %b.header

b.header:
  %x.b = load i32, i32* %ptr
  %v1 = call i32 @cond.i32()
  br label %c.header
; CHECK:       b.header:
; CHECK-NEXT:    %x.b = load i32, i32* %ptr
; CHECK-NEXT:    %v1 = call i32 @cond.i32()
; CHECK-NEXT:    switch i32 %v1, label %[[B_HEADER_SPLIT:.*]] [
; CHECK-NEXT:      i32 1, label %b.latch
; CHECK-NEXT:      i32 2, label %b.latch
; CHECK-NEXT:      i32 3, label %b.latch
; CHECK-NEXT:    ]
;
; CHECK:       [[B_HEADER_SPLIT]]:
; CHECK-NEXT:    %[[X_A_LCSSA:.*]] = phi i32 [ %x.a, %b.header ]
; CHECK-NEXT:    %[[X_B_LCSSA:.*]] = phi i32 [ %x.b, %b.header ]
; CHECK-NEXT:    br label %c.header

c.header:
  switch i32 %v1, label %c.latch [
    i32 1, label %b.latch
    i32 2, label %b.latch
    i32 3, label %b.latch
  ]
; CHECK:       c.header:
; CHECK-NEXT:    br label %c.latch

c.latch:
  ; Use values from other loops to check LCSSA form.
  store i32 %x.a, i32* %ptr
  store i32 %x.b, i32* %ptr
  %v2 = call i1 @cond()
  br i1 %v2, label %c.header, label %exit
; CHECK:       c.latch:
; CHECK-NEXT:    store i32 %[[X_A_LCSSA]], i32* %ptr
; CHECK-NEXT:    store i32 %[[X_B_LCSSA]], i32* %ptr
; CHECK-NEXT:    %v2 = call i1 @cond()
; CHECK-NEXT:    br i1 %v2, label %c.header, label %exit

b.latch:
  %v3 = call i1 @cond()
  br i1 %v3, label %b.header, label %a.latch
; CHECK:       b.latch:
; CHECK-NEXT:    %v3 = call i1 @cond()
; CHECK-NEXT:    br i1 %v3, label %b.header, label %a.latch

a.latch:
  br label %a.header
; CHECK:       a.latch:
; CHECK-NEXT:    br label %a.header

exit:
  ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
}

define void @test_unswitch_to_common_succ_with_phis(i32* %var, i32 %cond) {
; CHECK-LABEL: @test_unswitch_to_common_succ_with_phis(
entry:
  br label %header
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond, label %loopexit1 [
; CHECK-NEXT:      i32 13, label %loopexit2
; CHECK-NEXT:      i32 0, label %entry.split
; CHECK-NEXT:      i32 1, label %entry.split
; CHECK-NEXT:    ]
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %header

header:
  %var_val = load i32, i32* %var
  switch i32 %cond, label %loopexit1 [
    i32 0, label %latch
    i32 1, label %latch
    i32 13, label %loopexit2
  ]
; CHECK:       header:
; CHECK-NEXT:    load
; CHECK-NEXT:    br label %latch

latch:
  ; No-op PHI node to exercise weird PHI update scenarios.
  %phi = phi i32 [ %var_val, %header ], [ %var_val, %header ]
  call void @sink(i32 %phi)
  br label %header
; CHECK:       latch:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %var_val, %header ]
; CHECK-NEXT:    call void @sink(i32 %[[PHI]])
; CHECK-NEXT:    br label %header

loopexit1:
  ret void
; CHECK:       loopexit1:
; CHECK-NEXT:    ret

loopexit2:
  ret void
; CHECK:       loopexit2:
; CHECK-NEXT:    ret
}

define void @test_unswitch_to_default_common_succ_with_phis(i32* %var, i32 %cond) {
; CHECK-LABEL: @test_unswitch_to_default_common_succ_with_phis(
entry:
  br label %header
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 %cond, label %entry.split [
; CHECK-NEXT:      i32 13, label %loopexit
; CHECK-NEXT:    ]
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %header

header:
  %var_val = load i32, i32* %var
  switch i32 %cond, label %latch [
    i32 0, label %latch
    i32 1, label %latch
    i32 13, label %loopexit
  ]
; CHECK:       header:
; CHECK-NEXT:    load
; CHECK-NEXT:    br label %latch

latch:
  ; No-op PHI node to exercise weird PHI update scenarios.
  %phi = phi i32 [ %var_val, %header ], [ %var_val, %header ], [ %var_val, %header ]
  call void @sink(i32 %phi)
  br label %header
; CHECK:       latch:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %var_val, %header ]
; CHECK-NEXT:    call void @sink(i32 %[[PHI]])
; CHECK-NEXT:    br label %header

loopexit:
  ret void
; CHECK:       loopexit:
; CHECK-NEXT:    ret
}

declare void @f()
declare void @g()
define void @test_unswitch_switch_with_nonempty_unreachable() {
; CHECK-LABEL: @test_unswitch_switch_with_nonempty_unreachable()
entry:
  br label %loop

loop:
  %cleanup.dest.slot.0 = select i1 undef, i32 5, i32 undef
  br label %for.cond

for.cond:
  switch i32 %cleanup.dest.slot.0, label %NonEmptyUnreachableBlock [
    i32 0, label %for.cond
    i32 1, label %NonEmptyUnreachableBlock
    i32 2, label %loop.loopexit
  ]

loop.loopexit:
  unreachable

NonEmptyUnreachableBlock:
  call void @f()
  call void @g()
  unreachable

; CHECK:loop:
; CHECK-NEXT:  %cleanup.dest.slot.0 = select i1 undef, i32 5, i32 undef
; CHECK-NEXT:  switch i32 %cleanup.dest.slot.0, label %NonEmptyUnreachableBlock [
; CHECK-NEXT:    i32 1, label %NonEmptyUnreachableBlock
; CHECK-NEXT:    i32 0, label %loop.split
; CHECK-NEXT:    i32 2, label %loop.split
; CHECK-NEXT:  ]

; CHECK:loop.split:
; CHECK-NEXT:  br label %for.cond

; CHECK:for.cond:
; CHECK-NEXT:  switch i32 %cleanup.dest.slot.0, label %loop.loopexit [
; CHECK-NEXT:    i32 0, label %for.cond
; CHECK-NEXT:  ]

; CHECK:loop.loopexit:
; CHECK-NEXT:  unreachable

; CHECK:NonEmptyUnreachableBlock:
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  unreachable
}

define void @test_unswitch_switch_with_nonempty_unreachable2() {
; CHECK-LABEL: @test_unswitch_switch_with_nonempty_unreachable2()
entry:
  br label %loop

loop:
  %cleanup.dest.slot.0 = select i1 undef, i32 5, i32 undef
  br label %for.cond

for.cond:
  switch i32 %cleanup.dest.slot.0, label %for.cond [
    i32 0, label %for.cond
    i32 1, label %NonEmptyUnreachableBlock
    i32 2, label %loop.loopexit
  ]

loop.loopexit:
  unreachable

NonEmptyUnreachableBlock:
  call void @f()
  call void @g()
  unreachable

; CHECK:loop:
; CHECK-NEXT:  %cleanup.dest.slot.0 = select i1 undef, i32 5, i32 undef
; CHECK-NEXT:  switch i32 %cleanup.dest.slot.0, label %loop.split [
; CHECK-NEXT:    i32 1, label %NonEmptyUnreachableBlock
; CHECK-NEXT:  ]

; CHECK:loop.split:
; CHECK-NEXT:  br label %for.cond

; CHECK:for.cond:
; CHECK-NEXT:  switch i32 %cleanup.dest.slot.0, label %for.cond.backedge [
; CHECK-NEXT:    i32 0, label %for.cond.backedge
; CHECK-NEXT:    i32 2, label %loop.loopexit
; CHECK-NEXT:  ]

; CHECK:for.cond.backedge:
; CHECK-NEXT:  br label %for.cond

; CHECK:loop.loopexit:
; CHECK-NEXT:  unreachable

; CHECK:NonEmptyUnreachableBlock:
; CHECK-NEXT:  call void @f()
; CHECK-NEXT:  call void @g()
; CHECK-NEXT:  unreachable
}

; PR45355
define void @test_unswitch_switch_with_duplicate_edge() {
; CHECK-LABEL: @test_unswitch_switch_with_duplicate_edge()
entry:
  br label %lbl1

lbl1:                                             ; preds = %entry
  %cleanup.dest.slot.0 = select i1 undef, i32 5, i32 undef
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1, %lbl1
  switch i32 %cleanup.dest.slot.0, label %UnifiedUnreachableBlock [
    i32 0, label %for.cond1
    i32 5, label %UnifiedUnreachableBlock
    i32 2, label %lbl1.loopexit
  ]

UnifiedUnreachableBlock:                          ; preds = %for.cond1, %for.cond1
  unreachable

lbl1.loopexit:                                    ; preds = %for.cond1
  unreachable

; CHECK: for.cond1:
; CHECK-NEXT:  switch i32 %cleanup.dest.slot.0, label %UnifiedUnreachableBlock [
; CHECK-NEXT:    i32 0, label %for.cond1
; CHECK-NEXT:    i32 5, label %UnifiedUnreachableBlock
; CHECK-NEXT:    i32 2, label %lbl1.loopexit
; CHECK-NEXT:  ]
}
