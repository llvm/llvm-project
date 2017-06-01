; RUN: llvm-link -S %s %p/Inputs/comdat11.ll -o - | FileCheck %s

$foo = comdat any
@foo = global i8 0, comdat

; CHECK: @foo = global i8 0, comdat

; CHECK: define void @zed() {
; CHECK:   call void @bar()
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @bar()
