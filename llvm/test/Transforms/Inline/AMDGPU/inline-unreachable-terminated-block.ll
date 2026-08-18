; RUN: opt -mtriple=amdgcn-amd-amdhsa -inline-call-penalty=0 -inline-threshold=100 -passes=inline -S < %s | FileCheck %s --check-prefixes=CHECK,AMDGPU
; RUN: opt -inline-call-penalty=0 -inline-threshold=100 -passes=inline -S < %s | FileCheck %s --check-prefixes=CHECK,GENERIC

@g = external global i32

define void @callee() {
  store volatile i32 1, ptr @g
  ret void
}

define void @caller_unreachable() {
; CHECK-LABEL: @caller_unreachable
; AMDGPU:      store volatile i32 1, ptr @g
; AMDGPU-NOT:  call void @callee
; GENERIC:     call void @callee
  call void @callee()
  unreachable
}

define void @caller_reachable() {
; CHECK-LABEL: @caller_reachable
; CHECK-NOT:   call void @callee
  call void @callee()
  ret void
}
