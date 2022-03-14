; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-print-scops -disable-output < %s 2>&1 > /dev/null | FileCheck %s
;
; Error statements (%bb33) do not require their uses to be verified.
; In this case it uses %tmp32 from %bb31 which is not available because
; %bb31 is an error statement as well.

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

declare noalias i8* @widget()

declare void @quux()

define void @func(i32 %tmp3, i32 %tmp7, i32 %tmp17, i32 %tmp26, i32 %tmp19) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb5, label %bb16

bb5:                                              ; preds = %bb2
  %tmp8 = icmp eq i32 %tmp7, 0
  br i1 %tmp8, label %bb16, label %bb36

bb16:                                             ; preds = %bb5, %bb2
  %tmp18 = icmp eq i32 %tmp17, 0
  %tmp20 = icmp eq i32 %tmp19, 0
  %tmp21 = or i1 %tmp18, %tmp20
  br i1 %tmp21, label %bb31, label %bb25

bb25:                                             ; preds = %bb25, %bb16
  %tmp27 = icmp eq i32 %tmp26, 0
  br i1 %tmp27, label %bb31, label %bb25

bb31:                                             ; preds = %bb25, %bb16
  %tmp32 = call noalias i8* @widget()
  br label %bb33

bb33:                                             ; preds = %bb31
  call void @quux()
  %tmp34 = icmp eq i8* %tmp32, null
  br label %bb36

bb36:                                             ; preds = %bb33, %bb5
  ret void
}


; CHECK:      SCoP begins here.
; CHECK-NEXT: Low complexity assumption:       {  : false }
; CHECK-NEXT: SCoP ends here but was dismissed.
