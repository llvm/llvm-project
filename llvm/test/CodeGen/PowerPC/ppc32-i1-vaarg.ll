; RUN: llc -verify-machineinstrs < %s -mcpu=ppc32 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mcpu=ppc32 -mtriple=powerpc-darwin9 | FileCheck %s -check-prefix=CHECK-D
target triple = "powerpc-unknown-linux-gnu"

declare void @printf(i8*, ...)

define void @main() {
  call void (i8*, ...) @printf(i8* undef, i1 false)
  ret void
}

; CHECK-LABEL: @main
; CHECK-DAG: li 4, 0
; CHECK-DAG: crxor 6, 6, 6
; CHECK: bl printf

; CHECK-D-LABEL: @main
; CHECK-D: li r4, 0
; CHECK-D: bl _printf

