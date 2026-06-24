; RUN: llc < %s -mtriple=x86_64 -relocation-model=pic | FileCheck %s

; CHECK:      .globl  "\\\""
; CHECK-NEXT: "\\\"":
@"\\\22" = constant i8 0
