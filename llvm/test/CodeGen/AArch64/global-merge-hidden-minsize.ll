; RUN: llc %s -mtriple=aarch64-none-linux-gnu -o - | FileCheck %s

@x = hidden global i32 0, align 4
@y = hidden global i32 0, align 4

define hidden void @f() #0 {
  store i32 0, i32* @x, align 4
  store i32 0, i32* @y, align 4
  ret void
}

attributes #0 = { minsize optsize }

; CHECK: .local .L_MergedGlobals
; CHECK: .comm .L_MergedGlobals,8,4

; CHECK: .globl x
; CHECK: .hidden x
; CHECK: .set x, .L_MergedGlobals
; CHECK: .size x, 4

; CHECK: .globl y
; CHECK: .hidden y
; CHECK: .set y, .L_MergedGlobals+4
; CHECK: .size y, 4
