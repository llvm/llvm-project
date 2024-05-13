; RUN: llc -verify-machineinstrs < %s -mcpu=ppc32 | FileCheck %s
target triple = "powerpc-unknown-linux-gnu"

declare void @printf(ptr, ...)

define void @main() {
  call void (ptr, ...) @printf(ptr undef, i1 false)
  ret void
}

; CHECK-LABEL: @main
; CHECK-DAG: li 4, 0
; CHECK-DAG: crxor 6, 6, 6
; CHECK: bl printf


