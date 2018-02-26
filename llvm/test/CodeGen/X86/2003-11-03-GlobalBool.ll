; RUN: llc < %s -mtriple=i686-- | FileCheck %s

@X = global i1 true
; CHECK-NOT: .byte true
