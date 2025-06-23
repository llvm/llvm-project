; RUN: not llc < %s 2> %t.err | FileCheck %s
; RUN: FileCheck -check-prefix=ERR %s < %t.err
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".equiv pselect, __pselect"

; ERR: error: 'pselect' is a protected alias

; CHECK: .globl	pselect                         # -- Begin function pselect
; CHECK: .type	pselect,@function
; CHECK: .cfi_startproc # @pselect
; CHECK: retq
; CHECK: .size	pselect, .Lfunc_end0-pselect
define void @pselect() {
  ret void
}
