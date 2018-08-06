; RUN: llc -mtriple=aarch64-fuchsia -o - %s

define void @set_w18(i32 %x) {
entry:
; FIXME: Include an allocatable-specific error message
  tail call void @llvm.write_register.i32(metadata !0, i32 %x)
  ret void
}

declare void @llvm.write_register.i32(metadata, i32) nounwind

!0 = !{!"w18"}
