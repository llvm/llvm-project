; RUN: not --crash llc < %s -mtriple=aarch64-fuchsia 2>&1 | FileCheck %s

define void @set_x18(i64 %x) {
entry:
; FIXME: Include an allocatable-specific error message
; CHECK: Invalid register name "x18".
  tail call void @llvm.write_register.i64(metadata !0, i64 %x)
  ret void
}

declare void @llvm.write_register.i64(metadata, i64) nounwind

!0 = !{!"x18"}
