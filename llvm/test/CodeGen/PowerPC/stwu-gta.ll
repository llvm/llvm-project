target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-linux"
; RUN: llc -verify-machineinstrs < %s | FileCheck %s

%class.Two.0.5 = type { i32, i32, i32 }

@foo = external global %class.Two.0.5, align 4

define void @_GLOBAL__I_a() nounwind section ".text.startup" {
entry:
  store i32 5, ptr @foo, align 4
  store i32 6, ptr getelementptr inbounds (%class.Two.0.5, ptr @foo, i32 0, i32 1), align 4
  ret void
}

; CHECK: @_GLOBAL__I_a
; CHECK-NOT: stwux
; CHECK: stwu
