; REQUIRES: asserts
; RUN: opt -loop-unswitch -enable-new-pm=0 -disable-output -stats -info-output-file - < %s | FileCheck --check-prefix=STATS %s
; RUN: opt -S -loop-unswitch -enable-new-pm=0 -verify-loop-info -verify-dom-info -verify-memoryssa < %s | FileCheck %s

; STATS: 2 loop-unswitch - Number of switches unswitched

; CHECK:      %1 = icmp eq i32 %c, 1
; CHECK-NEXT: br i1 %1, label %.split.us, label %..split_crit_edge

; CHECK:      ..split_crit_edge:                                ; preds = %0
; CHECK-NEXT:   br label %.split

; CHECK:      .split.us:                                        ; preds = %0
; CHECK-NEXT:   br label %loop_begin.us

; CHECK:      loop_begin.us:                                    ; preds = %loop_begin.backedge.us, %.split.us
; CHECK-NEXT:   %var_val.us = load i32, i32* %var
; CHECK-NEXT:   switch i32 1, label %default.us-lcssa.us [
; CHECK-NEXT:     i32 1, label %inc.us

; CHECK:      inc.us:                                           ; preds = %loop_begin.us
; CHECK-NEXT:   call void @incf() [[NOR_NUW:#[0-9]+]]
; CHECK-NEXT:   br label %loop_begin.backedge.us

; CHECK:      .split:                                           ; preds = %..split_crit_edge
; CHECK-NEXT:   %2 = icmp eq i32 %c, 2
; CHECK-NEXT:   br i1 %2, label %.split.split.us, label %.split..split.split_crit_edge

; CHECK:      .split..split.split_crit_edge:                    ; preds = %.split
; CHECK-NEXT:   br label %.split.split

; CHECK:      .split.split.us:                                  ; preds = %.split
; CHECK-NEXT:   br label %loop_begin.us1

; CHECK:      loop_begin.us1:                                   ; preds = %loop_begin.backedge.us5, %.split.split.us
; CHECK-NEXT:   %var_val.us2 = load i32, i32* %var
; CHECK-NEXT:   switch i32 2, label %default.us-lcssa.us-lcssa.us [
; CHECK-NEXT:     i32 1, label %inc.split.us
; CHECK-NEXT:     i32 2, label %dec.us3
; CHECK-NEXT:   ]

; CHECK:      dec.us3:                                          ; preds = %loop_begin.us1
; CHECK-NEXT:   call void @decf() [[NOR_NUW]]
; CHECK-NEXT:   br label %loop_begin.backedge.us5

; CHECK:      .split.split:                                     ; preds = %.split..split.split_crit_edge
; CHECK-NEXT:   br label %loop_begin

; CHECK:      loop_begin:                                       ; preds = %loop_begin.backedge, %.split.split
; CHECK-NEXT:   %var_val = load i32, i32* %var
; CHECK-NEXT:   switch i32 %c, label %default.us-lcssa.us-lcssa [
; CHECK-NEXT:     i32 1, label %inc.split
; CHECK-NEXT:     i32 2, label %dec.split
; CHECK-NEXT:   ]

; CHECK:      inc.split:                                        ; preds = %loop_begin
; CHECK-NEXT:   br i1 true, label %us-unreachable.us-lcssa, label %inc

; CHECK:      dec.split:                                        ; preds = %loop_begin
; CHECK-NEXT:   br i1 true, label %us-unreachable6, label %dec

define i32 @test(i32* %var) {
  %mem = alloca i32
  store i32 2, i32* %mem
  %c = load i32, i32* %mem

  br label %loop_begin

loop_begin:

  %var_val = load i32, i32* %var

  switch i32 %c, label %default [
      i32 1, label %inc
      i32 2, label %dec
  ]

inc:
  call void @incf() noreturn nounwind
  br label %loop_begin
dec:
  call void @decf() noreturn nounwind
  br label %loop_begin
default:
  br label %loop_exit
loop_exit:
  ret i32 0
}

declare void @incf() noreturn
declare void @decf() noreturn

; CHECK: attributes #0 = { noreturn }
; CHECK: attributes [[NOR_NUW]] = { noreturn nounwind }
