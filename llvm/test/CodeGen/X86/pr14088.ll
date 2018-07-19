; RUN: llc -mtriple x86_64-linux -mcpu core2 -verify-machineinstrs %s -o - | FileCheck %s
define i32 @f(i1 %foo, i16* %tm_year2, i8* %bar, i16 %zed, i32 %zed2) {
entry:
  br i1 %foo, label %return, label %if.end

if.end:
  %rem = srem i32 %zed2, 100
  %conv3 = trunc i32 %rem to i16
  store i16 %conv3, i16* %tm_year2
  %sext = shl i32 %rem, 16
  %conv5 = ashr exact i32 %sext, 16
  %div = sdiv i32 %conv5, 10
  %conv6 = trunc i32 %div to i8
  store i8 %conv6, i8* %bar
  br label %return

return:
  %retval.0 = phi i32 [ 0, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}

; We were miscompiling this and using %ax instead of %cx in the movw
; in the following sequence:
;	movswl	%cx, %ecx
;	movw	%cx, (%rsi)
;	movslq	%ecx, %rcx
;
; We can't produce the above sequence without special SD-level
; heuristics. Now we produce this:
; CHECK: movw	%ax, (%rsi)
; CHECK: cwtl
; CHECK: cltq
