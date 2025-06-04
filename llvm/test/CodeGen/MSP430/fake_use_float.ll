; RUN: llc -stop-after=finalize-isel -filetype=asm -o - < %s | FileCheck %s
target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430"

; CHECK:       bb.0.entry:
; CHECK:       %0:gr16 = MOV16ri
; CHECK-DAG:   FAKE_USE killed %0
; CHECK:       %1:gr16 = MOV16ri
; CHECK-DAG:   FAKE_USE killed %1
; CHECK:       %2:gr16 = MOV16ri
; CHECK-DAG:   FAKE_USE killed %2
; CHECK:       %3:gr16 = MOV16ri
; CHECK-DAG:   FAKE_USE killed %3
; CHECK:       RET
define void @test-double() {
entry:
  call void (...) @llvm.fake.use(double -8.765430e+02)
  ret void
}

; CHECK:       bb.0.entry:
; CHECK:       %0:gr16 = MOV16ri
; CHECK-DAG:   FAKE_USE killed %0
; CHECK:       %1:gr16 = MOV16ri
; CHECK-DAG:   FAKE_USE killed %1
; CHECK:       RET
define void @test-float() {
entry:
  call void (...) @llvm.fake.use(float -8.76e+02)
  ret void
}
