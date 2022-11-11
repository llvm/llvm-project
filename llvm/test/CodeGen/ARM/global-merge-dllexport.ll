; RUN: llc < %s -mtriple=thumbv7-win32 -arm-global-merge | FileCheck %s

@x = global i32 0, align 4
@y = dllexport global i32 0, align 4

define void @f1(i32 %a1, i32 %a2) {
; CHECK: f1:
; CHECK: movw [[REG1:r[0-9]+]], :lower16:.L_MergedGlobals
; CHECK: movt [[REG1]], :upper16:.L_MergedGlobals
  store i32 %a1, i32* @x, align 4
  store i32 %a2, i32* @y, align 4
  ret void
}

; CHECK: .lcomm .L_MergedGlobals,8,4
; CHECK: .section .drectve,"yni"
; CHECK: .ascii " /EXPORT:y,DATA"
; CHECK: .globl x
; CHECK: .set x, .L_MergedGlobals
; CHECK: .globl y
; CHECK: .set y, .L_MergedGlobals+4
