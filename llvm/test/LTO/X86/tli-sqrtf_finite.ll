; RUN: llvm-as < %s > %t
; RUN: llvm-lto %t -exported-symbol=foo -filetype=asm -o - | FileCheck %s

; Check that sqrtf_finite is recognized as a libcall by SelectionDAGBuilder
; to enable sqrtss instruction to be used.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @foo(float %x) {
; CHECK: sqrtss
  %call = tail call nnan ninf float @__sqrtf_finite(float %x) readnone
  ret float %call
}

declare float @__sqrtf_finite(float) readnone
