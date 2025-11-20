; RUN: llc -mtriple=sparc64-linux-gnu -mattr=+reserve-l0 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVED-L0

;; Ensure explicit register references are catched as well.

; CHECK-RESERVED-L0: %l0
define void @set_reg(i32 zeroext %x) {
entry:
  tail call void @llvm.write_register.i32(metadata !0, i32 %x)
  ret void
}

declare void @llvm.write_register.i32(metadata, i32)
!0 = !{!"l0"}
