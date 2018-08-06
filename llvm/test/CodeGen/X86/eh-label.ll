; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s
; Test that we don't crashe if the .Lfunc_end0 name is taken.

declare void @g()

define void @f() personality i8* bitcast (void ()* @g to i8*) {
bb0:
  call void asm ".Lfunc_end0:", ""()
; CHECK: #APP
; CHECK-NEXT: .Lfunc_end0:
; CHECK-NEXT: #NO_APP

  invoke void @g() to label %bb2 unwind label %bb1
bb1:
  landingpad { i8*, i32 }
          catch i8* null
  call void @g()
  ret void
bb2:
  ret void

; CHECK: [[END:.Lfunc_end.*]]:
; CHECK: .uleb128	[[END]]-
}
