; RUN: not --crash llc -mtriple=m68k -mattr=+reserve-a0 < %s 2>&1 | FileCheck %s

; CHECK: Named registers not implemented for this target
define noundef i32 @foo() {
  tail call void @llvm.write_register.i32(metadata !0, i32 321)
  ret i32 0
}

declare void @llvm.write_register.i32(metadata, i32)

!llvm.named.register.a0 = !{!0}
!0 = !{!"d2"}
