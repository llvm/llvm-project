; RUN: llc < %s -march=x86 | FileCheck %s

@X = global i1 true
; CHECK-NOT: .byte true
