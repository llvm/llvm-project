; RUN: llc -mtriple=aarch64-fuchsia -o - %s

define void @set_x18(i64 %x) {
entry:
; FIXME: Include an allocatable-specific error message
  tail call void @llvm.write_register.i64(metadata !0, i64 %x)
  ret void
}

declare void @llvm.write_register.i64(metadata, i64) nounwind

!0 = !{!"x18"}
