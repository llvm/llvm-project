; RUN: opt < %s -deadargelim -S | FileCheck %s

; rdar://11546243
%struct.A = type { i8 }

define available_externally void @_Z17externallyDefinedP1A(%struct.A* %a) {
entry:
  call void @_Z3foov()
  ret void
}

declare void @_Z3foov()

define void @_Z4testP1A(%struct.A* %a) {
; CHECK: @_Z4testP1A
; CHECK: @_Z17externallyDefinedP1A(%struct.A* %a)

entry:
  call void @_Z17externallyDefinedP1A(%struct.A* %a)
  ret void
}
