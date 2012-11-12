; RUN: llvm-as < %s | llvm-dis > %t0
; RUN: opt -S < %s > %t1
; RUN: diff %t0 %t1
; PR6140

; Make sure the flags are serialized/deserialized properly for both
; forward and backward references.

define void @foo() nounwind {
entry:
  br label %first

second:                                           ; preds = %first
  %u = add nuw i32 %a, 0                          ; <i32> [#uses=0]
  %s = add nsw i32 %a, 0                          ; <i32> [#uses=0]
  %us = add nuw nsw i32 %a, 0                     ; <i32> [#uses=0]
  %z = add i32 %a, 0                              ; <i32> [#uses=0]
  unreachable

first:                                            ; preds = %entry
  %a = bitcast i32 0 to i32                       ; <i32> [#uses=8]
  %uu = add nuw i32 %a, 0                         ; <i32> [#uses=0]
  %ss = add nsw i32 %a, 0                         ; <i32> [#uses=0]
  %uuss = add nuw nsw i32 %a, 0                   ; <i32> [#uses=0]
  %zz = add i32 %a, 0                             ; <i32> [#uses=0]
  br label %second
}
