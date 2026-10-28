; RUN: llc -mtriple=sparc64-linux-gnu -mattr=+reserve-l0 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVED-L0
; RUN: llc -mtriple=sparc64-linux-gnu -mattr=+reserve-l0 -stop-after=finalize-isel -o - %s | FileCheck %s


;; Ensure explicit register references are catched as well.
; CHECK-RESERVED-L0: %l0
define void @set_reg(i32 zeroext %x) {
entry:
  tail call void @llvm.write_register.i32(metadata !0, i32 %x)
  ret void
}

define i64 @get_sp() nounwind {
; CHECK-LABEL: name: get_sp
; CHECK:       {{%[0-9]+}}:i64regs = COPY $o6
  %1 = call i64 @llvm.read_register.i64(metadata !1)
  ret i64 %1
}

define i64 @get_fp() nounwind {
; CHECK-LABEL: name: get_fp
; CHECK:       {{%[0-9]+}}:i64regs = COPY $i6
entry:
  %1 = call i64 @llvm.read_register.i64(metadata !2)
  ret i64 %1
}

!llvm.named.register = !{!0, !1, !2}
declare void @llvm.write_register.i32(metadata, i32)
!0 = !{!"%l0"}
!1 = !{!"%sp"}
!2 = !{!"%fp"}
