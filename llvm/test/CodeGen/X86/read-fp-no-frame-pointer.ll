; RUN: not llc < %s -mtriple=x86_64-linux-gnueabi 2>&1 | FileCheck %s

define i32 @get_frame() nounwind {
entry:
; CHECK: register ebp is allocatable: function has no frame pointer
  %fp = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %fp
}

declare i32 @llvm.read_register.i32(metadata) nounwind

!0 = !{!"ebp\00"}
