; RUN: opt < %s -loop-unswitch -loop-unswitch-threshold=0 -verify-loop-info -S < %s 2>&1 | FileCheck %s
; RUN: opt < %s -loop-unswitch -loop-unswitch-threshold=0 -verify-loop-info -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s 2>&1 | FileCheck %s

; This test contains two trivial unswitch condition in one loop. 
; LoopUnswitch pass should be able to unswitch the second one 
; after unswitching the first one.


; CHECK:  br i1 %cond1, label %..split_crit_edge, label %.loop_exit.split_crit_edge

; CHECK:  ..split_crit_edge:                                ; preds = %0
; CHECK:    br label %.split

; CHECK:  .split:                                           ; preds = %..split_crit_edge
; CHECK:    br i1 %cond2, label %.split..split.split_crit_edge, label %.split.loop_exit.split1_crit_edge

; CHECK:  .split..split.split_crit_edge:                    ; preds = %.split
; CHECK:    br label %.split.split

; CHECK:  .split.split:                                     ; preds = %.split..split.split_crit_edge
; CHECK:    br label %loop_begin

; CHECK:  loop_begin:                                       ; preds = %do_something, %.split.split
; CHECK:    br i1 true, label %continue, label %loop_exit

; CHECK:  continue:                                         ; preds = %loop_begin
; CHECK:    %var_val = load i32, i32* %var
; CHECK:    br i1 true, label %do_something, label %loop_exit

define i32 @test(i32* %var, i1 %cond1, i1 %cond2) {
  br label %loop_begin

loop_begin:  
  br i1 %cond1, label %continue, label %loop_exit	; first trivial condition

continue:
  %var_val = load i32, i32* %var
  br i1 %cond2, label %do_something, label %loop_exit	; second trivial condition  

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}


; We will not be able trivially unswitch on the SwitchInst, as its input
; is a constant. However, since its a constant we should be able to figure
; out that the switch can be folded into a unconditional branch to %continue.
; Then we unswitch on the br inst in %continue.
;
; CHECK: define i32 @test2(
; This is an indication that the loop has been unswitched on %cond1.
; CHECK:  br i1 %cond1, label %..split_crit_edge, label %.loop_exit.split_crit_edge

; CHECK:  ..split_crit_edge:                                ; preds = %0
; CHECK:    br label %.split

; CHECK:  .split:                                           ; preds = %..split_crit_edge
; CHECK:    br label %loop_begin

; CHECK:  loop_begin:                                       ; preds = %do_something, %.split
; CHECK:    switch i32

; CHECK:  continue:                                         ; preds = %loop_begin
; CHECK:    %var_val = load i32, i32* %var
; CHECK:    br i1 true, label %do_something, label %loop_exit

define i32 @test2(i32* %var, i1 %cond1) {
  br label %loop_begin

loop_begin:  
  switch i32 1, label %continue [
    i32 0, label %loop_exit
    i32 1, label %continue
  ]

continue:
  %var_val = load i32, i32* %var
  br i1 %cond1, label %do_something, label %loop_exit

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

declare void @some_func() noreturn
