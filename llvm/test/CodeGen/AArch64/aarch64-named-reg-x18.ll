; RUN: not llc -mtriple=aarch64-fuchsia -o - %s 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -mtriple=aarch64-fuchsia -mattr=+reserve-x18 -o - %s

define void @set_x18(i64 %x) {
entry:
; FIXME: Include an allocatable-specific error message
; ERROR: Invalid register name "x18".
  tail call void @llvm.write_register.i64(metadata !0, i64 %x)
  ret void
}

declare void @llvm.write_register.i64(metadata, i64) nounwind

!0 = !{!"x18"}
