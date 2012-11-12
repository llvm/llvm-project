; RUN: opt -loop-unswitch -loop-unswitch-threshold 1000 -disable-output -stats -info-output-file - < %s | FileCheck --check-prefix=STATS %s
; RUN: opt -S -loop-unswitch -loop-unswitch-threshold 1000 -verify-loop-info -verify-dom-info %s | FileCheck %s

; STATS: 1 loop-simplify - Number of pre-header or exit blocks inserted
; STATS: 3 loop-unswitch - Number of switches unswitched

; CHECK:        %1 = icmp eq i32 %c, 1
; CHECK-NEXT:   br i1 %1, label %.split.us, label %..split_crit_edge

; CHECK:      ..split_crit_edge:                                ; preds = %0
; CHECK-NEXT:   br label %.split

; CHECK:      .split.us:                                        ; preds = %0
; CHECK-NEXT:   %2 = icmp eq i32 %d, 1
; CHECK-NEXT:   br i1 %2, label %.split.us.split.us, label %.split.us..split.us.split_crit_edge

; CHECK:      .split.us..split.us.split_crit_edge:              ; preds = %.split.us
; CHECK-NEXT:   br label %.split.us.split

; CHECK:      .split.us.split.us:                               ; preds = %.split.us
; CHECK-NEXT:   br label %loop_begin.us.us

; CHECK:      loop_begin.us.us:                                 ; preds = %loop_begin.backedge.us.us, %.split.us.split.us
; CHECK-NEXT:   %var_val.us.us = load i32* %var
; CHECK-NEXT:   switch i32 1, label %second_switch.us.us [
; CHECK-NEXT:     i32 1, label %inc.us.us

; CHECK:      second_switch.us.us:                              ; preds = %loop_begin.us.us
; CHECK-NEXT:   switch i32 1, label %default.us.us [
; CHECK-NEXT:     i32 1, label %inc.us.us

; CHECK:      inc.us.us:                                        ; preds = %second_switch.us.us, %loop_begin.us.us
; CHECK-NEXT:   call void @incf() noreturn nounwind
; CHECK-NEXT:   br label %loop_begin.backedge.us.us

; CHECK:      .split.us.split:                                  ; preds = %.split.us..split.us.split_crit_edge
; CHECK-NEXT:   br label %loop_begin.us

; CHECK:      loop_begin.us:                                    ; preds = %loop_begin.backedge.us, %.split.us.split
; CHECK-NEXT:   %var_val.us = load i32* %var
; CHECK-NEXT:   switch i32 1, label %second_switch.us [
; CHECK-NEXT:     i32 1, label %inc.us

; CHECK:      second_switch.us:                                 ; preds = %loop_begin.us
; CHECK-NEXT:   switch i32 %d, label %default.us [
; CHECK-NEXT:     i32 1, label %second_switch.us.inc.us_crit_edge
; CHECK-NEXT:   ]

; CHECK:      second_switch.us.inc.us_crit_edge:                ; preds = %second_switch.us
; CHECK-NEXT:   br i1 true, label %us-unreachable8, label %inc.us

; CHECK:      inc.us:                                           ; preds = %second_switch.us.inc.us_crit_edge, %loop_begin.us
; CHECK-NEXT:   call void @incf() noreturn nounwind
; CHECK-NEXT:   br label %loop_begin.backedge.us

; CHECK:      .split:                                           ; preds = %..split_crit_edge
; CHECK-NEXT:   %3 = icmp eq i32 %d, 1
; CHECK-NEXT:   br i1 %3, label %.split.split.us, label %.split..split.split_crit_edge

; CHECK:      .split..split.split_crit_edge:                    ; preds = %.split
; CHECK-NEXT:   br label %.split.split

; CHECK:      .split.split.us:                                  ; preds = %.split
; CHECK-NEXT:   br label %loop_begin.us1

; CHECK:      loop_begin.us1:                                   ; preds = %loop_begin.backedge.us6, %.split.split.us
; CHECK-NEXT:   %var_val.us2 = load i32* %var
; CHECK-NEXT:   switch i32 %c, label %second_switch.us3 [
; CHECK-NEXT:     i32 1, label %loop_begin.inc_crit_edge.us
; CHECK-NEXT:   ]

; CHECK:      second_switch.us3:                                ; preds = %loop_begin.us1
; CHECK-NEXT:   switch i32 1, label %default.us5 [
; CHECK-NEXT:     i32 1, label %inc.us4
; CHECK-NEXT:   ]

; CHECK:      inc.us4:                                          ; preds = %loop_begin.inc_crit_edge.us, %second_switch.us3
; CHECK-NEXT:   call void @incf() noreturn nounwind
; CHECK-NEXT:   br label %loop_begin.backedge.us6

; CHECK:      loop_begin.inc_crit_edge.us:                      ; preds = %loop_begin.us1
; CHECK-NEXT:   br i1 true, label %us-unreachable.us-lcssa.us, label %inc.us4

; CHECK:      .split.split:                                     ; preds = %.split..split.split_crit_edge
; CHECK-NEXT:   br label %loop_begin

; CHECK:      loop_begin:                                       ; preds = %loop_begin.backedge, %.split.split
; CHECK-NEXT:   %var_val = load i32* %var
; CHECK-NEXT:   switch i32 %c, label %second_switch [
; CHECK-NEXT:     i32 1, label %loop_begin.inc_crit_edge
; CHECK-NEXT:   ]

; CHECK:      loop_begin.inc_crit_edge:                         ; preds = %loop_begin
; CHECK-NEXT:   br i1 true, label %us-unreachable.us-lcssa, label %inc

; CHECK:      second_switch:                                    ; preds = %loop_begin
; CHECK-NEXT:   switch i32 %d, label %default [
; CHECK-NEXT:     i32 1, label %second_switch.inc_crit_edge
; CHECK-NEXT:   ]

; CHECK:      second_switch.inc_crit_edge:                      ; preds = %second_switch
; CHECK-NEXT:   br i1 true, label %us-unreachable7, label %inc


define i32 @test(i32* %var) {
  %mem = alloca i32
  store i32 2, i32* %mem
  %c = load i32* %mem
  %d = load i32* %mem

  br label %loop_begin

loop_begin:

  %var_val = load i32* %var

  switch i32 %c, label %second_switch [
      i32 1, label %inc
  ]

second_switch:
  switch i32 %d, label %default [
      i32 1, label %inc
  ]

inc:
  call void @incf() noreturn nounwind
  br label %loop_begin

default:
  br label %loop_begin

loop_exit:
  ret i32 0
}

declare void @incf() noreturn
declare void @decf() noreturn
