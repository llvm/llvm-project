; Make sure @main is left untouched.
; RUN: opt -metarenamer -S %s | FileCheck %s
; RUN: opt -passes=metarenamer -S %s | FileCheck %s

; CHECK: define void @main
; CHECK: call void @main

define void @main() {
  call void @patatino()
  ret void
}

define void @patatino() {
  call void @main()
  ret void
}
