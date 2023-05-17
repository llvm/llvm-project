; RUN: llc < %s
; PR6777

; MachineSink shouldn't try to sink code in unreachable blocks, as it's
; not worthwhile, and there are corner cases which it doesn't handle.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define double @fn1(ptr %arg, i64 %arg1) {
Entry:
  br i1 undef, label %Body, label %Exit

Exit:                                             ; preds = %Brancher7, %Entry
  ret double undef

Body:                                             ; preds = %Entry
  br i1 false, label %Brancher7, label %Body3

Body3:                                            ; preds = %Body6, %Body3, %Body
  br label %Body3

Body6:                                            ; preds = %Brancher7
  %tmp = fcmp oeq double 0xC04FBB2E40000000, undef ; <i1> [#uses=1]
  br i1 %tmp, label %Body3, label %Brancher7

Brancher7:                                        ; preds = %Body6, %Body
  %tmp2 = icmp ult i32 undef, 10                  ; <i1> [#uses=1]
  br i1 %tmp2, label %Body6, label %Exit
}
