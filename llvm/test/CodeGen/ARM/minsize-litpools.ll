; RUN: llc -mtriple=thumbv7s %s -o -  | FileCheck %s
; RUN: llc -mtriple=armv7s %s -o -  | FileCheck %s

; CodeGen should be able to set and reset the MinSize subtarget-feature, and
; make use of it in deciding whether to use MOVW/MOVT for global variables or a
; lit-pool load (saving roughly 2 bytes of code).

@var = global i32 0

define i32 @small_global() minsize {
; CHECK-LABEL: small_global:
; CHECK: ldr r[[GLOBDEST:[0-9]+]], {{.?LCPI0_0}}
; CHECK: ldr r0, [r[[GLOBDEST]]]

  %val = load i32, ptr @var
  ret i32 %val
}

define i32 @big_global() {
; CHECK-LABEL: big_global:
; CHECK: movw [[GLOBDEST:r[0-9]+]], :lower16:var
; CHECK: movt [[GLOBDEST]], :upper16:var

  %val = load i32, ptr @var
  ret i32 %val
}
