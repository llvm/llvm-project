; RUN: llc --show-mc-encoding < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define i32 @f0(i32* nocapture %x) nounwind readonly ssp {
entry:
  %tmp1 = load i32* %x                            ; <i32> [#uses=2]
  %tobool = icmp eq i32 %tmp1, 0                  ; <i1> [#uses=1]
  br i1 %tobool, label %if.end, label %return

if.end:                                           ; preds = %entry

; Check that we lower to the short form of cmpl, which has a fixed %eax
; register.
;
; CHECK: cmpl $16777216, %eax
; CHECK: # encoding: [0x3d,0x00,0x00,0x00,0x01]
  %cmp = icmp eq i32 %tmp1, 16777216              ; <i1> [#uses=1]

  %conv = zext i1 %cmp to i32                     ; <i32> [#uses=1]
  ret i32 %conv

return:                                           ; preds = %entry
  ret i32 0
}
