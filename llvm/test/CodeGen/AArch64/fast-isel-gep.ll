; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

%struct.foo = type { i32, i64, float, double }

define double* @test_struct(%struct.foo* %f) {
; CHECK-LABEL: test_struct
; CHECK:       add x0, x0, #24
  %1 = getelementptr inbounds %struct.foo, %struct.foo* %f, i64 0, i32 3
  ret double* %1
}

define i32* @test_array1(i32* %a, i64 %i) {
; CHECK-LABEL: test_array1
; CHECK:       orr [[REG:x[0-9]+]], xzr, #0x4
; CHECK-NEXT:  madd  x0, x1, [[REG]], x0
  %1 = getelementptr inbounds i32, i32* %a, i64 %i
  ret i32* %1
}

define i32* @test_array2(i32* %a) {
; CHECK-LABEL: test_array2
; CHECK:       add  x0, x0, #16
  %1 = getelementptr inbounds i32, i32* %a, i64 4
  ret i32* %1
}

define i32* @test_array3(i32* %a) {
; CHECK-LABEL: test_array3
; CHECK:       add x0, x0, #1, lsl #12
  %1 = getelementptr inbounds i32, i32* %a, i64 1024
  ret i32* %1
}

define i32* @test_array4(i32* %a) {
; CHECK-LABEL: test_array4
; CHECK:       mov [[REG:x[0-9]+]], #4104
; CHECK-NEXR:  add x0, x0, [[REG]]
  %1 = getelementptr inbounds i32, i32* %a, i64 1026
  ret i32* %1
}

define i32* @test_array5(i32* %a, i32 %i) {
; CHECK-LABEL: test_array5
; CHECK:       sxtw [[REG1:x[0-9]+]], w1
; CHECK-NEXT:  orr  [[REG2:x[0-9]+]], xzr, #0x4
; CHECK-NEXT:  madd  {{x[0-9]+}}, [[REG1]], [[REG2]], x0
  %1 = getelementptr inbounds i32, i32* %a, i32 %i
  ret i32* %1
}
