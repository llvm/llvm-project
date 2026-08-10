; RUN: llc < %s | FileCheck %s
; rdar://8051990

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11"

; ISel doesn't yet know how to eliminate this extra zero-extend. But until
; it knows how to do so safely, it shouldn;t eliminate it.
; CHECK: movzbl  (%rdi), %eax
; CHECK: movzwl  %ax, %eax

define i64 @_ZL5matchPKtPKhiR9MatchData(ptr %tmp13) nounwind {
entry:
  %tmp14 = load i8, ptr %tmp13, align 1
  %tmp17 = zext i8 %tmp14 to i16
  br label %bb341

bb341:
  %tmp18 = add i16 %tmp17, -1
  %tmp23 = sext i16 %tmp18 to i64
  ret i64 %tmp23
}
