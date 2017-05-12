; RUN: opt -passes='loop(unswitch),verify<loops>' -S < %s | FileCheck %s

declare void @some_func() noreturn

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
; CHECK-NEXT:    switch i32 %cond2, label %[[UNREACHABLE:.*]] [
; CHECK-NEXT:      i32 0, label %loop0
; CHECK-NEXT:      i32 1, label %loop1
; CHECK-NEXT:      i32 2, label %loop2
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
;
; CHECK:       [[UNREACHABLE]]:
; CHECK-NEXT:    unreachable
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
