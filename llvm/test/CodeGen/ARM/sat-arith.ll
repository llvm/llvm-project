; RUN: llc -O1 -mtriple=armv6-none-none-eabi %s -o - | FileCheck %s -check-prefix=ARM -check-prefix=CHECK
; RUN: llc -O1 -mtriple=thumbv7-none-none-eabi %s -o - | FileCheck %s -check-prefix=THUMB -check-prefix=CHECK

; CHECK-LABEL: qadd
define i32 @qadd() nounwind {
; CHECK-DAG: mov{{s?}} [[R0:.*]], #8
; CHECK-DAG: mov{{s?}} [[R1:.*]], #128
; CHECK-ARM: qadd [[R0]], [[R1]], [[R0]]
; CHECK-THRUMB: qadd [[R0]], [[R0]], [[R1]]
  %tmp = call i32 @llvm.arm.qadd(i32 128, i32 8)
  ret i32 %tmp
}

; CHECK-LABEL: qsub
define i32 @qsub() nounwind {
; CHECK-DAG: mov{{s?}} [[R0:.*]], #8
; CHECK-DAG: mov{{s?}} [[R1:.*]], #128
; CHECK-ARM: qsub [[R0]], [[R1]], [[R0]]
; CHECK-THRUMB: qadd [[R0]], [[R1]], [[R0]]
  %tmp = call i32 @llvm.arm.qsub(i32 128, i32 8)
  ret i32 %tmp
}

; upper-bound of the immediate argument
; CHECK-LABEL: ssat1
define i32 @ssat1() nounwind {
; CHECK: mov{{s?}} [[R0:.*]], #128
; CHECK: ssat [[R1:.*]], #32, [[R0]]
  %tmp = call i32 @llvm.arm.ssat(i32 128, i32 32)
  ret i32 %tmp
}

; lower-bound of the immediate argument
; CHECK-LABEL: ssat2
define i32 @ssat2() nounwind {
; CHECK: mov{{s?}} [[R0:.*]], #128
; CHECK: ssat [[R1:.*]], #1, [[R0]]
  %tmp = call i32 @llvm.arm.ssat(i32 128, i32 1)
  ret i32 %tmp
}

; upper-bound of the immediate argument
; CHECK-LABEL: usat1
define i32 @usat1() nounwind {
; CHECK: mov{{s?}} [[R0:.*]], #128
; CHECK: usat [[R1:.*]], #31, [[R0]]
  %tmp = call i32 @llvm.arm.usat(i32 128, i32 31)
  ret i32 %tmp
}

; lower-bound of the immediate argument
; CHECK-LABEL: usat2
define i32 @usat2() nounwind {
; CHECK: mov{{s?}} [[R0:.*]], #128
; CHECK: usat [[R1:.*]], #0, [[R0]]
  %tmp = call i32 @llvm.arm.usat(i32 128, i32 0)
  ret i32 %tmp
}

declare i32 @llvm.arm.qadd(i32, i32) nounwind
declare i32 @llvm.arm.qsub(i32, i32) nounwind
declare i32 @llvm.arm.ssat(i32, i32) nounwind readnone
declare i32 @llvm.arm.usat(i32, i32) nounwind readnone
