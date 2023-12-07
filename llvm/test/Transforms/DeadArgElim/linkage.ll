; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

; rdar://11546243
%struct.A = type { i8 }

define available_externally void @_Z17externallyDefinedP1A(ptr %a) {
entry:
  call void @_Z3foov()
  ret void
}

declare void @_Z3foov()

define void @_Z4testP1A(ptr %a) {
; CHECK: @_Z4testP1A
; CHECK: @_Z17externallyDefinedP1A(ptr %a)

entry:
  call void @_Z17externallyDefinedP1A(ptr %a)
  ret void
}
