; RUN: llc < %s -mattr=+simd128 | FileCheck %s

; Regression test for a crash caused by
; WebAssemblyTargetLowering::ReplaceNodeResults not being
; implemented. Since SIMD is enabled, sign_ext_inreg is custom lowered
; but the result is i16, an illegal value. This requires
; ReplaceNodeResults to resolve, but the default implementation is to
; abort.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

; CHECK: i32.load8_s
; CHECK-NEXT: i32.store16
define void @foo() {
entry:
  %0 = load i32*, i32** undef, align 4
  %1 = load i32, i32* %0, align 4
  %2 = load i32, i32* undef, align 4
  %conv67 = trunc i32 %2 to i8
  %conv68 = sext i8 %conv67 to i16
  store i16 %conv68, i16* null, align 2
  ret void
}
