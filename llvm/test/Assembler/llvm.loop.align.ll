; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Valid "llvm.loop.align" metadata round-trips through the assembler.

define void @pow2() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i32 64}

define void @one() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !2
exit:
  ret void
}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.align", i32 1}

; CHECK: [[LOOP0:![0-9]+]] = distinct !{[[LOOP0]], [[ALIGN0:![0-9]+]]}
; CHECK: [[ALIGN0]] = !{!"llvm.loop.align", i32 64}
; CHECK: [[LOOP2:![0-9]+]] = distinct !{[[LOOP2]], [[ALIGN2:![0-9]+]]}
; CHECK: [[ALIGN2]] = !{!"llvm.loop.align", i32 1}
