; RUN: llc < %s | FileCheck %s
; Don't try to emit a direct call through a TLS global.
; This fixes PR22103

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external thread_local global i64

; Function Attrs: nounwind
define void @_Z1fv() {
; CHECK-NOT: callq *$a
; CHECK: movq %fs:0, [[RAX:%r..]]
; CHECK-NEXT: addq    a@GOTTPOFF(%rip), [[RAX]]
; CHECK-NEXT: callq *[[RAX]]
entry:
  call void @a()
  ret void
}
