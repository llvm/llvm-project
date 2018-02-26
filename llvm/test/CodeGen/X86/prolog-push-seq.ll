; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc18.0.0"

declare x86_thiscallcc void @bar(i32 %a, i32 %b)

define fastcc void @foo(i32 %a, i32 %b) #0 {
; CHECK-LABEL: foo:
; CHECK: subl $64, %esp
; CHECK-NEXT: pushl
; CHECK-NEXT: calll _bar
  %local = alloca i32, i32 16
  call x86_thiscallcc void @bar(i32 %a, i32 %b)
  call void asm sideeffect "nop", "~{ax},~{bx},~{cx},~{dx},~{bp},~{si},~{di}"()
  ret void
}

attributes #0 = { nounwind optsize "no-frame-pointer-elim-non-leaf"}
