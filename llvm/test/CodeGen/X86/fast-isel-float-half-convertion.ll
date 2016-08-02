; RUN: llc -fast-isel -fast-isel-abort=1 -asm-verbose=false -mtriple=x86_64-unknown-unknown -mattr=+f16c < %s | FileCheck %s

; Verify that fast-isel correctly expands float-half conversions.

define i16 @test_fp32_to_fp16(float %a) {
; CHECK-LABEL: test_fp32_to_fp16:
; CHECK: vcvtps2ph $0, %xmm0, %xmm0
; CHECK-NEXT: vmovd %xmm0, %eax
; CHECK-NEXT: retq
entry:
  %0 = call i16 @llvm.convert.to.fp16.f32(float %a)
  ret i16 %0
}

define float @test_fp16_to_fp32(i32 %a) {
; CHECK-LABEL: test_fp16_to_fp32:
; CHECK: movswl %di, %eax
; CHECK-NEXT: vmovd %eax, %xmm0
; CHECK-NEXT: vcvtph2ps %xmm0, %xmm0
; CHECK-NEXT: retq
entry:
  %0 = trunc i32 %a to i16
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)
  ret float %1
}

declare i16 @llvm.convert.to.fp16.f32(float)
declare float @llvm.convert.from.fp16.f32(i16)
