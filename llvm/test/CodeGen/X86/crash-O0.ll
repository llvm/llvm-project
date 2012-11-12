; RUN: llc -O0 -relocation-model=pic -disable-fp-elim < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10"

; This file contains functions that may crash llc -O0

; The DIV8 instruction produces results in AH and AL, but we don't want to use
; AH in 64-bit mode. The hack used must not generate copyFromReg nodes for
; aliased registers (AX and AL) - RegAllocFast does not like that.
; PR7312
define i32 @div8() nounwind {
entry:
  %0 = trunc i64 undef to i8                      ; <i8> [#uses=3]
  %1 = udiv i8 0, %0                              ; <i8> [#uses=1]
  %2 = urem i8 0, %0                              ; <i8> [#uses=1]
  %3 = icmp uge i8 %2, %0                         ; <i1> [#uses=1]
  br i1 %3, label %"40", label %"39"

"39":                                             ; preds = %"36"
  %4 = zext i8 %1 to i32                          ; <i32> [#uses=1]
  %5 = mul nsw i32 %4, undef                      ; <i32> [#uses=1]
  %6 = add nsw i32 %5, undef                      ; <i32> [#uses=1]
  %7 = icmp ne i32 %6, undef                      ; <i1> [#uses=1]
  br i1 %7, label %"40", label %"41"

"40":                                             ; preds = %"39", %"36"
  unreachable

"41":                                             ; preds = %"39"
  unreachable
}
