; RUN: not llc < %s -mtriple=arm64-apple-darwin 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=arm64-linux-gnueabi 2>&1 | FileCheck %s

define i32 @get_stack() nounwind {
entry:
; CHECK: Invalid register name "notareg".
	%sp = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %sp
}

declare i32 @llvm.read_register.i32(metadata) nounwind

!0 = !{!"notareg\00"}
