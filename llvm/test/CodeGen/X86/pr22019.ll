; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm "pselect = __pselect"
module asm "var = __var"
module asm "alias = __alias"
; CHECK: pselect = __pselect
; CHECK: var = __var
; CHECK: alias = __alias

; CHECK: pselect:
; CHECK: retq
define void @pselect() {
  ret void
}

; CHECK: var:
; CHECK: .long 0
@var = global i32 0

; CHECK: alias = var
@alias = alias i32, i32* @var
