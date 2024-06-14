; RUN: llc -mcpu=yonah < %s
; PR9438
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-unknown-freebsd9.0"

; The 'call fastcc' ties down %ebx, %ecx, and %edx.
; A MUL8r ties down %al, leaving no GR32_ABCD registers available.
; The coalescer can easily overallocate physical registers,
; and register allocation fails.

declare fastcc ptr @save_string(ptr %d, ptr nocapture %s) nounwind

define i32 @cvtchar(ptr nocapture %sp) nounwind {
  %temp.i = alloca [2 x i8], align 1
  %tmp1 = load i8, ptr %sp, align 1
  %div = udiv i8 %tmp1, 10
  %rem = urem i8 %div, 10
  store i8 %rem, ptr %temp.i, align 1
  %call.i = call fastcc ptr @save_string(ptr %sp, ptr %temp.i) nounwind
  ret i32 undef
}
