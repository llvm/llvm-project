; RUN: llc -mtriple=arm-unknown-unknown < %s | FileCheck %s

; CHECK: test1
define float @test1() nounwind uwtable readnone ssp {
; CHECK-NOT: floorf
  %foo = call float @floorf(float 0x4000CCCCC0000000) nounwind readnone
  ret float %foo
}

; CHECK: test2
define float @test2() nounwind uwtable readnone ssp {
; CHECK-NOT: ceilf
  %foo = call float @ceilf(float 0x4000CCCCC0000000) nounwind readnone
  ret float %foo
}

; CHECK: test3
define float @test3() nounwind uwtable readnone ssp {
; CHECK-NOT: truncf
  %foo = call float @truncf(float 0x4000CCCCC0000000) nounwind readnone
  ret float %foo
}

declare float @floorf(float) nounwind readnone
declare float @ceilf(float) nounwind readnone
declare float @truncf(float) nounwind readnone



