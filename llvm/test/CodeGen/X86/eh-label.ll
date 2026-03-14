; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s
; Test that we don't crashe if the .Lfunc_end0 name is taken.

declare void @g()

define void @f() personality ptr @g {
bb0:
  call void asm ".Lfunc_end0:", ""()
; CHECK: #APP
; CHECK-NEXT: .Lfunc_end0:
; CHECK-NEXT: #NO_APP

  invoke void @g() to label %bb2 unwind label %bb1
bb1:
  landingpad { ptr, i32 }
          catch ptr null
  call void @g()
  ret void
bb2:
  ret void

; CHECK: [[END:.Lfunc_end.*]]:
; CHECK: .uleb128	[[END]]-
}
