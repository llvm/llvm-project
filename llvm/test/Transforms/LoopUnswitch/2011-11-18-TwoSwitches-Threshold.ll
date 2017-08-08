; REQUIRES: asserts
; RUN: opt -loop-unswitch -loop-unswitch-threshold 13 -disable-output -stats -info-output-file - < %s | FileCheck --check-prefix=STATS %s
; RUN: opt -S -loop-unswitch -loop-unswitch-threshold 13 -verify-loop-info -verify-dom-info < %s | FileCheck %s

; STATS: 1 loop-unswitch - Number of switches unswitched

; ModuleID = '../llvm/test/Transforms/LoopUnswitch/2011-11-18-TwoSwitches.ll'

; CHECK:        %1 = icmp eq i32 %c, 1
; CHECK-NEXT:   br i1 %1, label %.split.us, label %..split_crit_edge

; CHECK:      ..split_crit_edge:                                ; preds = %0
; CHECK-NEXT:   br label %.split

; CHECK:      .split.us:                                        ; preds = %0
; CHECK-NEXT:   br label %loop_begin.us

; CHECK:      loop_begin.us:                                    ; preds = %loop_begin.backedge.us, %.split.us
; CHECK:        switch i32 1, label %second_switch.us [
; CHECK-NEXT:     i32 1, label %inc.us

; CHECK:      second_switch.us:                                 ; preds = %loop_begin.us
; CHECK-NEXT:   switch i32 %d, label %default.us [
; CHECK-NEXT:     i32 1, label %inc.us
; CHECK-NEXT:   ]

; CHECK:      inc.us:                                           ; preds = %second_switch.us, %loop_begin.us
; CHECK-NEXT:   call void @incf() [[NOR_NUW:#[0-9]+]]
; CHECK-NEXT:   br label %loop_begin.backedge.us

; CHECK:      .split:                                           ; preds = %..split_crit_edge
; CHECK-NEXT:   br label %loop_begin

; CHECK:      loop_begin:                                       ; preds = %loop_begin.backedge, %.split
; CHECK:        switch i32 %c, label %second_switch [
; CHECK-NEXT:     i32 1, label %loop_begin.inc_crit_edge
; CHECK-NEXT:   ]

; CHECK:      loop_begin.inc_crit_edge:                         ; preds = %loop_begin
; CHECK-NEXT:   br i1 true, label %us-unreachable, label %inc

; CHECK:      second_switch:                                    ; preds = %loop_begin
; CHECK-NEXT:   switch i32 %d, label %default [
; CHECK-NEXT:     i32 1, label %inc
; CHECK-NEXT:   ]

; CHECK:      inc:                                              ; preds = %loop_begin.inc_crit_edge, %second_switch
; CHECK-NEXT:   call void @incf() [[NOR_NUW]]
; CHECK-NEXT:   br label %loop_begin.backedge

define i32 @test(i32* %var) {
  %mem = alloca i32
  store i32 2, i32* %mem
  %c = load i32, i32* %mem
  %d = load i32, i32* %mem

  br label %loop_begin

loop_begin:

  %var_val = load i32, i32* %var

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

; CHECK: attributes #0 = { noreturn }
; CHECK: attributes [[NOR_NUW]] = { noreturn nounwind }
