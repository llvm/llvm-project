; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK-LABEL: f0Test
; CHECK: entsp 1
; CHECK: bl f0
; CHECK: retsp 1
%struct.st0 = type { [0 x i32] }
declare void @f0(ptr) nounwind
define void @f0Test(ptr byval(%struct.st0) %s0) nounwind {
entry:
  call void @f0(ptr %s0) nounwind
  ret void
}

; CHECK-LABEL: f1Test
; CHECK: entsp 13
; CHECK: stw r4, sp[12]
; CHECK: stw r5, sp[11]
; CHECK: mov r4, r0
; CHECK: ldaw r5, sp[1]
; CHECK: ldc r2, 40
; CHECK: mov r0, r5
; CHECK: bl __memcpy_4
; CHECK: mov r0, r5
; CHECK: bl f1
; CHECK: mov r0, r4
; CHECK: ldw r5, sp[11]
; CHECK: ldw r4, sp[12]
; CHECK: retsp 13
%struct.st1 = type { [10 x i32] }
declare void @f1(ptr) nounwind
define i32 @f1Test(i32 %i, ptr byval(%struct.st1) %s1) nounwind {
entry:
  call void @f1(ptr %s1) nounwind
  ret i32 %i
}

; CHECK-LABEL: f2Test
; CHECK: extsp 4
; CHECK: stw lr, sp[1]
; CHECK: mov r11, r1
; CHECK-DAG: stw r2, sp[3]
; CHECK-DAG: stw r3, sp[4]
; CHECK: ldw r0, r0[0]
; CHECK: stw r0, sp[2]
; CHECK: ldaw r1, sp[2]
; CHECK: mov r0, r11
; CHECK: bl f2
; CHECK: ldw lr, sp[1]
; CHECK: ldaw sp, sp[4]
; CHECK: retsp 0
%struct.st2 = type { i32 }
declare void @f2(i32, ptr) nounwind
define void @f2Test(ptr byval(%struct.st2) %s2, i32 %i, ...) nounwind {
entry:
  call void @f2(i32 %i, ptr %s2)
  ret void
}

; CHECK-LABEL: f3Test
; CHECK: entsp 2
; CHECK: ldc r1, 0
; CHECK: ld8u r2, r0[r1]
; CHECK: ldaw r0, sp[1]
; CHECK: st8 r2, r0[r1]
; CHECK: bl f
; CHECK: retsp 2
declare void @f3(ptr) nounwind
define void @f3Test(ptr byval(i8) %v) nounwind {
entry:
  call void @f3(ptr %v) nounwind
  ret void
}
