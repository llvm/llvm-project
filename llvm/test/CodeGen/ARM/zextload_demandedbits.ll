; RUN: llc < %s -mtriple="arm-apple-ios3.0.0" | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"

%struct.eggs = type { %struct.spam, i16 }
%struct.spam = type { [3 x i32] }
%struct.barney = type { [2 x i32], [2 x i32] }

; Make sure that the sext op does not get lost due to computeKnownBits.
; CHECK: quux
; CHECK: lsl
; CHECK: asr
; CHECK: bl
; CHECK: pop
define void @quux(%struct.eggs* %arg) {
bb:
  %tmp1 = getelementptr inbounds %struct.eggs, %struct.eggs* %arg, i32 0, i32 1
  %0 = load i16, i16* %tmp1, align 2
  %tobool = icmp eq i16 %0, 0
  br i1 %tobool, label %bb16, label %bb3

bb3:                                              ; preds = %bb
  %tmp4 = bitcast i16* %tmp1 to i8*
  %tmp5 = ptrtoint i16* %tmp1 to i32
  %tmp6 = shl i32 %tmp5, 20
  %tmp7 = ashr exact i32 %tmp6, 20
  %tmp14 = getelementptr inbounds %struct.barney, %struct.barney* undef, i32 %tmp7
  %tmp15 = tail call i32 @widget(%struct.barney* %tmp14, i8* %tmp4, i32 %tmp7)
  br label %bb16

bb16:                                             ; preds = %bb3, %bb
  ret void
}

declare i32 @widget(%struct.barney*, i8*, i32)
