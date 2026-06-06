; RUN: llc < %s -mtriple=armv8-linux-gnueabihf -mattr=+fp-armv8 | FileCheck --check-prefix=CHECK --check-prefix=DP %s
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabihf -mattr=+fp-armv8,-d32,-fp64 | FileCheck --check-prefix=SP %s
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabihf -mattr=+fp-armv8,-d32 | FileCheck --check-prefix=DP %s

; CHECK-LABEL: test1
; CHECK: vrintm.f32
define float @test1(float %a) {
entry:
  %call = call float @llvm.floor.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test2
; SP: b floor
; DP: vrintm.f64
define double @test2(double %a) {
entry:
  %call = call double @llvm.floor.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test3
; CHECK: vrintp.f32
define float @test3(float %a) {
entry:
  %call = call float @llvm.ceil.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test4
; SP: b ceil
; DP: vrintp.f64
define double @test4(double %a) {
entry:
  %call = call double @llvm.ceil.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test5
; CHECK: vrinta.f32
define float @test5(float %a) {
entry:
  %call = call float @llvm.round.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test6
; SP: b round
; DP: vrinta.f64
define double @test6(double %a) {
entry:
  %call = call double @llvm.round.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test7
; CHECK: vrintz.f32
define float @test7(float %a) {
entry:
  %call = call float @llvm.trunc.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test8
; SP: b trunc
; DP: vrintz.f64
define double @test8(double %a) {
entry:
  %call = call double @llvm.trunc.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test9
; CHECK: vrintr.f32
define float @test9(float %a) {
entry:
  %call = call float @llvm.nearbyint.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test10
; SP: b nearbyint
; DP: vrintr.f64
define double @test10(double %a) {
entry:
  %call = call double @llvm.nearbyint.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test11
; CHECK: vrintx.f32
define float @test11(float %a) {
entry:
  %call = call float @llvm.rint.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test12
; SP: b rint
; DP: vrintx.f64
define double @test12(double %a) {
entry:
  %call = call double @llvm.rint.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test13
; CHECK: vrintn.f32
define float @test13(float %a) {
entry:
  %round = call float @llvm.roundeven.f32(float %a)
  ret float %round
}

; CHECK-LABEL: test14
; CHECK: vrintn.f64
define double @test14(double %a) {
entry:
  %round = call double @llvm.roundeven.f64(double %a)
  ret double %round
}

declare float @llvm.floor.f32(float) nounwind readnone
declare double @llvm.floor.f64(double) nounwind readnone
declare float @llvm.ceil.f32(float) nounwind readnone
declare double @llvm.ceil.f64(double) nounwind readnone
declare float @llvm.round.f32(float) nounwind readnone
declare double @llvm.round.f64(double) nounwind readnone
declare float @llvm.trunc.f32(float) nounwind readnone
declare double @llvm.trunc.f64(double) nounwind readnone
declare float @llvm.rint.f32(float) nounwind readnone
declare double @llvm.rint.f64(double) nounwind readnone
declare float @llvm.roundeven.f32(float)
declare double @llvm.roundeven.f64(double)
