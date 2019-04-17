; RUN: opt < %s -globalopt -S | FileCheck %s

; CHECK-NOT: aa
; CHECK-NOT: bb

declare void @aa()
@bb = external global i8
