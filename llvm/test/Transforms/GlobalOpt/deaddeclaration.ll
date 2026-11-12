; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; CHECK-NOT: aa
; CHECK-NOT: bb

declare void @aa()
@bb = external global i8
