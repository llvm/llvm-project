; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu \
; RUN:          -mattr=+spe |  FileCheck %s

define i32 @test_f32(float %x) {
; CHECK-LABEL: test_f32:
; CHECK:         #APP
; CHECK-NEXT:    efsctsi 31, 3
; CHECK-NEXT:    #NO_APP
entry:
  %0 = call i32 asm sideeffect "efsctsi $0, $1", "={f31},f"(float %x)
  ret i32 %0
}

define i32 @test_f64(double %x) {
; CHECK-LABEL: test_f64:
; CHECK:         #APP
; CHECK-NEXT:    efdctsi 0, 3
; CHECK-NEXT:    #NO_APP
entry:
  %0 = call i32 asm sideeffect "efdctsi $0, $1", "={f0},d"(double %x)
  ret i32 %0
}

