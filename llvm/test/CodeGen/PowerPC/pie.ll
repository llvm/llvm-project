; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-pc-linux -relocation-model=pic | FileCheck %s


define dso_local void @f() {
  ret void
}

define dso_local void @g() {
; CHECK: g:
; CHECK: bl f{{$}}
  call void @f()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIE Level", i32 1}
