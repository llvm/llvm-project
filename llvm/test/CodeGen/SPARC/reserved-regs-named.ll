; RUN: llc -mtriple=sparc64-linux-gnu -mattr=+reserve-l0 -verify-machineinstrs -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVED-L0
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

define i64 @get_r0() nounwind {
; CHECK-LABEL: name: get_r0
; CHECK:       {{%[0-9]+}}:i64regs = COPY $g0
  %1 = call i64 @llvm.read_register.i64(metadata !3)
  ret i64 %1
}

define i64 @get_r14() nounwind {
; CHECK-LABEL: name: get_r14
; CHECK:       {{%[0-9]+}}:i64regs = COPY $o6
  %1 = call i64 @llvm.read_register.i64(metadata !4)
  ret i64 %1
}

define i64 @get_r16() nounwind {
; CHECK-LABEL: name: get_r16
; CHECK:       {{%[0-9]+}}:i64regs = COPY $l0
  %1 = call i64 @llvm.read_register.i64(metadata !5)
  ret i64 %1
}

define i64 @get_r30() nounwind {
; CHECK-LABEL: name: get_r30
; CHECK:       {{%[0-9]+}}:i64regs = COPY $i6
  %1 = call i64 @llvm.read_register.i64(metadata !6)
  ret i64 %1
}

define i64 @get_r31() nounwind {
; CHECK-LABEL: name: get_r31
; CHECK:       {{%[0-9]+}}:i64regs = COPY $i7
  %1 = call i64 @llvm.read_register.i64(metadata !7)
  ret i64 %1
}

!llvm.named.register = !{!0, !1, !2, !3, !4, !5, !6, !7}
declare void @llvm.write_register.i32(metadata, i32)
!0 = !{!"%l0"}
!1 = !{!"%sp"}
!2 = !{!"%fp"}
!3 = !{!"r0"}
!4 = !{!"%r14"}
!5 = !{!"r16"}
!6 = !{!"%r30"}
!7 = !{!"r31"}
