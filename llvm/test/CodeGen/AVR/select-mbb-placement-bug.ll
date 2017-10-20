; RUN: llc -mcpu=atmega328p < %s -march=avr | FileCheck %s

; CHECK-LABEL: loopy
define internal fastcc void @loopy() {

; In this case, when we expand `Select8`/`Select16`, we should be
; replacing the existing MBB instead of adding a new one.
;
; https://github.com/avr-rust/rust/issues/49

; CHECK: LBB0_{{[0-9]+}}:
; CHECK: LBB0_{{[0-9]+}}:
; CHECK-NOT: LBB0_{{[0-9]+}}:
start:
  br label %bb7.preheader

bb7.preheader:                                    ; preds = %bb10, %start
  %i = phi i8 [ 0, %start ], [ %j, %bb10 ]
  %j = phi i8 [ 1, %start ], [ %next, %bb10 ]
  br label %bb10

bb4:                                              ; preds = %bb10
  ret void

bb10:                                             ; preds = %bb7.preheader
  tail call fastcc void @observe(i8 %i, i8 1)
  %0 = icmp ult i8 %j, 20
  %1 = zext i1 %0 to i8
  %next = add i8 %j, %1
  br i1 %0, label %bb7.preheader, label %bb4

}

declare void @observe(i8, i8);

