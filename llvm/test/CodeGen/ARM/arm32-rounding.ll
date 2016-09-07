; RUN: llc < %s -mtriple=armv8-linux-gnueabihf -mattr=+fp-armv8 | FileCheck --check-prefix=CHECK --check-prefix=DP %s
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabihf -mattr=+fp-armv8,+d16,+fp-only-sp | FileCheck --check-prefix=SP %s
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabihf -mattr=+fp-armv8,+d16 | FileCheck --check-prefix=DP %s

; CHECK-LABEL: test1
; CHECK: vrintm.f32
define float @test1(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test2
; SP: b floor
; DP: vrintm.f64
define double @test2(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test3
; CHECK: vrintp.f32
define float @test3(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test4
; SP: b ceil
; DP: vrintp.f64
define double @test4(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test5
; CHECK: vrinta.f32
define float @test5(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test6
; SP: b round
; DP: vrinta.f64
define double @test6(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test7
; CHECK: vrintz.f32
define float @test7(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test8
; SP: b trunc
; DP: vrintz.f64
define double @test8(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test9
; CHECK: vrintr.f32
define float @test9(float %a) {
entry:
  %call = call float @nearbyintf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test10
; SP: b nearbyint
; DP: vrintr.f64
define double @test10(double %a) {
entry:
  %call = call double @nearbyint(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test11
; CHECK: vrintx.f32
define float @test11(float %a) {
entry:
  %call = call float @rintf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test12
; SP: b rint
; DP: vrintx.f64
define double @test12(double %a) {
entry:
  %call = call double @rint(double %a) nounwind readnone
  ret double %call
}

declare float @floorf(float) nounwind readnone
declare double @floor(double) nounwind readnone
declare float @ceilf(float) nounwind readnone
declare double @ceil(double) nounwind readnone
declare float @roundf(float) nounwind readnone
declare double @round(double) nounwind readnone
declare float @truncf(float) nounwind readnone
declare double @trunc(double) nounwind readnone
declare float @nearbyintf(float) nounwind readnone
declare double @nearbyint(double) nounwind readnone
declare float @rintf(float) nounwind readnone
declare double @rint(double) nounwind readnone
