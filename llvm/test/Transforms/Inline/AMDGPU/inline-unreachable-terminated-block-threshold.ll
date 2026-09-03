; RUN: opt -mtriple=amdgcn-amd-amdhsa -inline-call-penalty=0 -inline-threshold=1 -passes=inline -S < %s | FileCheck %s

@g = external global i32

define void @small() {
  store volatile i32 1, ptr @g
  ret void
}

define void @big() {
  store volatile i32 1, ptr @g
  store volatile i32 2, ptr @g
  store volatile i32 3, ptr @g
  store volatile i32 4, ptr @g
  store volatile i32 5, ptr @g
  ret void
}

define void @caller_small() {
; CHECK-LABEL: @caller_small
; CHECK:       store volatile i32 1, ptr @g
; CHECK-NOT:   call void @small
  call void @small()
  unreachable
}

define void @caller_big() {
; CHECK-LABEL: @caller_big
; CHECK:       call void @big
  call void @big()
  unreachable
}
