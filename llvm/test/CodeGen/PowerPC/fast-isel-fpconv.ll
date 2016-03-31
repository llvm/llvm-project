; RUN: llc -mtriple powerpc64-unknown-linux-gnu -fast-isel -O0 < %s | FileCheck %s

; The second fctiwz would use an incorrect input register due to wrong handling
; of COPY_TO_REGCLASS in the FastISel pass.  Verify that this is fixed.

declare void @func(i32, i32)

define void @test() {
; CHECK-LABEL: test:
; CHECK: bl func
; CHECK-NEXT: nop
; CHECK: lfs [[REG:[0-9]+]], 
; CHECK: fctiwz {{[0-9]+}}, [[REG]]
; CHECK: bl func
; CHECK-NEXT: nop

  %memPos = alloca float, align 4
  store float 1.500000e+01, float* %memPos
  %valPos = load float, float* %memPos

  %memNeg = alloca float, align 4
  store float -1.500000e+01, float* %memNeg
  %valNeg = load float, float* %memNeg

  %FloatToIntPos = fptosi float %valPos to i32
  call void @func(i32 15, i32 %FloatToIntPos)

  %FloatToIntNeg = fptosi float %valNeg to i32
  call void @func(i32 -15, i32 %FloatToIntNeg)

  ret void
}

