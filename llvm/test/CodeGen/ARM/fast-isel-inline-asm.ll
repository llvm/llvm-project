; RUN: llc -opaque-pointers=0 -fast-isel < %s | FileCheck %s
target datalayout = "e-m:o-p:32:32-i1:8:32-i8:8:32-i16:16:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

%0 = type opaque

; Make sure that the inline asm starts right after the call to bar.
define void @test_inline_asm_sideeffect(%0* %call) {
; CHECK:      bl _bar
; CHECK-NEXT: InlineAsm Start
  call void @bar()
  call void asm sideeffect "mov\09r7, r7\09\09@ marker", ""()
  %1 = call %0* bitcast (i8* (i8*)* @foo to %0* (%0*)*)(%0* %call)
  ret void
}

declare i8* @foo(i8*)
declare void @bar()
