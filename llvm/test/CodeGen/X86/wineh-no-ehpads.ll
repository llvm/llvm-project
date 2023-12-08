; RUN: llc < %s | FileCheck %s

target triple = "x86_64-pc-windows-msvc"

declare void @g()
declare i32 @__CxxFrameHandler3(...)

define void @personality_no_ehpad() personality ptr @__CxxFrameHandler3 {
  call void @g()
  ret void
}

; CHECK-LABEL: personality_no_ehpad: # @personality_no_ehpad
; CHECK-NOT: movq $-2,
; CHECK: callq g
; CHECK: nop
; CHECK: retq

; Shouldn't have any LSDA either.
; CHECK-NOT: cppxdata
