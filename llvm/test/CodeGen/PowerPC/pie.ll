; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-pc-linux -relocation-model=pic | FileCheck %s


define void @f() {
  ret void
}

define void @g() {
; CHECK: g:
; CHECK: bl f{{$}}
  call void @f()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIE Level", i32 1}
