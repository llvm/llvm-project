; RUN: llc < %s -verify-machineinstrs | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; This test exercises the alias checking in SimpleRegisterCoalescing::RemoveCopyByCommutingDef.

define void @f(ptr %w, ptr %h, ptr %_this, ptr %image) nounwind ssp {
  %x1 = tail call i64 @g(ptr %_this, ptr %image) nounwind ; <i64> [#uses=3]
  %tmp1 = trunc i64 %x1 to i32                     ; <i32> [#uses=1]
; CHECK: movl (%r{{.*}}), %
  %x4 = load i32, ptr %h, align 4                      ; <i32> [#uses=1]

; The imull clobbers a 32-bit register.
; CHECK: imull %{{...}}, %e[[CLOBBER:..]]
  %x5 = mul nsw i32 %x4, %tmp1                      ; <i32> [#uses=1]

; So we cannot use the corresponding 64-bit register anymore.
; CHECK-NOT: shrq $32, %r[[CLOBBER]]
  %btmp3 = lshr i64 %x1, 32                         ; <i64> [#uses=1]
  %btmp4 = trunc i64 %btmp3 to i32                  ; <i32> [#uses=1]

; CHECK: idiv
  %x6 = sdiv i32 %x5, %btmp4                         ; <i32> [#uses=1]
  store i32 %x6, ptr %w, align 4
  ret void
}

declare i64 @g(ptr, ptr)
