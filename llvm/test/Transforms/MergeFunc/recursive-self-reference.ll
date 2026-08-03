; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; Recursive call targets may still be considered equivalent.
; CHECK-LABEL: define internal void @recursive_f(
; CHECK:         call void @recursive_f()
; CHECK-NOT:     @recursive_g
; CHECK-LABEL: define i32 @main(
; CHECK:         call void @recursive_f()
; CHECK:         call void @recursive_f()

define internal void @recursive_f() {
  call void @recursive_f()
  ret void
}

define internal void @recursive_g() {
  call void @recursive_g()
  ret void
}

define i32 @main() {
  call void @recursive_f()
  call void @recursive_g()
  ret i32 0
}
