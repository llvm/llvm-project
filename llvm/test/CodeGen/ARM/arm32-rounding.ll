; RUN: llc < %s -mtriple=armv8-linux-gnueabihf -mattr=+fp-armv8 | FileCheck --check-prefix=CHECK --check-prefix=DP %s
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabihf -mattr=+fp-armv8,+d16,+fp-only-sp | FileCheck --check-prefix=SP %s
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabihf -mattr=+fp-armv8,+d16 | FileCheck --check-prefix=DP %s

; CHECK-LABEL: test1
; CHECK: vrintm.f32
define arm_aapcs_vfpcc float @test1(float %a) {
entry:
  %call = call arm_aapcs_vfpcc float @floorf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test2
; SP: bl floor
; DP: vrintm.f64
define arm_aapcs_vfpcc double @test2(double %a) {
entry:
  %call = call arm_aapcscc double @floor(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test3
; CHECK: vrintp.f32
define arm_aapcs_vfpcc float @test3(float %a) {
entry:
  %call = call arm_aapcs_vfpcc float @ceilf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test4
; SP: bl ceil
; DP: vrintp.f64
define arm_aapcs_vfpcc double @test4(double %a) {
entry:
  %call = call arm_aapcscc double @ceil(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test5
; CHECK: vrinta.f32
define arm_aapcs_vfpcc float @test5(float %a) {
entry:
  %call = call arm_aapcs_vfpcc float @roundf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test6
; SP: bl round
; DP: vrinta.f64
define arm_aapcs_vfpcc double @test6(double %a) {
entry:
  %call = call arm_aapcscc double @round(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test7
; CHECK: vrintz.f32
define arm_aapcs_vfpcc float @test7(float %a) {
entry:
  %call = call arm_aapcs_vfpcc float @truncf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test8
; SP: bl trunc
; DP: vrintz.f64
define arm_aapcs_vfpcc double @test8(double %a) {
entry:
  %call = call arm_aapcscc double @trunc(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test9
; CHECK: vrintr.f32
define arm_aapcs_vfpcc float @test9(float %a) {
entry:
  %call = call arm_aapcs_vfpcc float @nearbyintf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test10
; SP: bl nearbyint
; DP: vrintr.f64
define arm_aapcs_vfpcc double @test10(double %a) {
entry:
  %call = call arm_aapcscc double @nearbyint(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test11
; CHECK: vrintx.f32
define arm_aapcs_vfpcc float @test11(float %a) {
entry:
  %call = call arm_aapcs_vfpcc float @rintf(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test12
; SP: bl rint
; DP: vrintx.f64
define arm_aapcs_vfpcc double @test12(double %a) {
entry:
  %call = call arm_aapcscc double @rint(double %a) nounwind readnone
  ret double %call
}

declare arm_aapcs_vfpcc float @floorf(float) nounwind readnone
declare arm_aapcscc double @floor(double) nounwind readnone
declare arm_aapcs_vfpcc float @ceilf(float) nounwind readnone
declare arm_aapcscc double @ceil(double) nounwind readnone
declare arm_aapcs_vfpcc float @roundf(float) nounwind readnone
declare arm_aapcscc double @round(double) nounwind readnone
declare arm_aapcs_vfpcc float @truncf(float) nounwind readnone
declare arm_aapcscc double @trunc(double) nounwind readnone
declare arm_aapcs_vfpcc float @nearbyintf(float) nounwind readnone
declare arm_aapcscc double @nearbyint(double) nounwind readnone
declare arm_aapcs_vfpcc float @rintf(float) nounwind readnone
declare arm_aapcscc double @rint(double) nounwind readnone

