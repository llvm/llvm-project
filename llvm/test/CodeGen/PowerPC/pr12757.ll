; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @__flt_rounds() nounwind {
entry:
  %0 = tail call i64 asm sideeffect "mffs $0", "=f"() nounwind
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK: @__flt_rounds
; CHECK: mffs

