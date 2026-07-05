; RUN: llc -mtriple=arm-unknown-unknown < %s | FileCheck %s

; CHECK: test1
define float @test1() nounwind uwtable readnone ssp {
; CHECK-NOT: floorf
  %foo = call float @llvm.floor.f32(float 0x4000CCCCC0000000) nounwind readnone
  ret float %foo
}

; CHECK: test2
define float @test2() nounwind uwtable readnone ssp {
; CHECK-NOT: ceilf
  %foo = call float @llvm.ceil.f32(float 0x4000CCCCC0000000) nounwind readnone
  ret float %foo
}

; CHECK: test3
define float @test3() nounwind uwtable readnone ssp {
; CHECK-NOT: truncf
  %foo = call float @llvm.trunc.f32(float 0x4000CCCCC0000000) nounwind readnone
  ret float %foo
}

declare float @llvm.floor.f32(float) nounwind readnone
declare float @llvm.ceil.f32(float) nounwind readnone
declare float @llvm.trunc.f32(float) nounwind readnone



