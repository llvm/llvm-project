; RUN: llc < %s -march=xcore | FileCheck %s

define void @_Z1fz(...) {
entry:
; CHECK-LABEL: _Z1fz:
; CHECK: extsp 3
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp[[[REG]]]
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp[[[REG]]]
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp[[[REG]]]
; CHECK: stw r[[REG:[0-3]{1,1}]]
; CHECK: , sp[[[REG]]]
; CHECK: ldaw sp, sp[3]
; CHECK: retsp 0
  ret void
}


declare void @llvm.va_start(ptr) nounwind
declare void @llvm.va_end(ptr) nounwind
declare void @f(i32) nounwind
define void @test_vararg(...) nounwind {
entry:
; CHECK-LABEL: test_vararg
; CHECK: extsp 6
; CHECK: stw lr, sp[1]
; CHECK-DAG: stw r3, sp[6]
; CHECK-DAG: stw r0, sp[3]
; CHECK-DAG: stw r1, sp[4]
; CHECK-DAG: stw r2, sp[5]
; CHECK: ldaw r0, sp[3]
; CHECK: stw r0, sp[2]
  %list = alloca ptr, align 4
  call void @llvm.va_start(ptr %list)
  br label %for.cond

; CHECK-LABEL: .LBB1_1
; CHECK: ldw r0, sp[2]
; CHECK: add r1, r0, 4
; CHECK: stw r1, sp[2]
; CHECK: ldw r0, r0[0]
; CHECK: bl f
; CHECK: bu .LBB1_1
for.cond:
  %0 = va_arg ptr %list, i32
  call void @f(i32 %0)
  br label %for.cond

  call void @llvm.va_end(ptr %list)
  ret void
}

