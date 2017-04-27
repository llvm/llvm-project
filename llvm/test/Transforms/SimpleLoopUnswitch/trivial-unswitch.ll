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
