; RUN: not llc -mtriple=aarch64-fuchsia -o - %s 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -mtriple=aarch64-fuchsia -mattr=+reserve-x18 -o - %s

define void @set_w18(i32 %x) {
entry:
; FIXME: Include an allocatable-specific error message
; ERROR: Invalid register name "w18".
  tail call void @llvm.write_register.i32(metadata !0, i32 %x)
  ret void
}

declare void @llvm.write_register.i32(metadata, i32) nounwind

!0 = !{!"w18"}
