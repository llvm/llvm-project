; RUN: llc -O3 < %s -mtriple=arm64 | FileCheck %s

; CHECK-LABEL: test1:
; CHECK: frintm
; CHECK-NOT: frintx
define float @test1(float %a) #0 {
entry:
  %call = tail call float @llvm.floor.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test2:
; CHECK: frintm
; CHECK-NOT: frintx
define double @test2(double %a) #0 {
entry:
  %call = tail call double @llvm.floor.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test3:
; CHECK: frinti
define float @test3(float %a) #0 {
entry:
  %call = tail call float @llvm.nearbyint.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test4:
; CHECK: frinti
define double @test4(double %a) #0 {
entry:
  %call = tail call double @llvm.nearbyint.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test5:
; CHECK: frintp
; CHECK-NOT: frintx
define float @test5(float %a) #0 {
entry:
  %call = tail call float @llvm.ceil.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test6:
; CHECK: frintp
; CHECK-NOT: frintx
define double @test6(double %a) #0 {
entry:
  %call = tail call double @llvm.ceil.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test7:
; CHECK: frintx
define float @test7(float %a) #0 {
entry:
  %call = tail call float @llvm.rint.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test8:
; CHECK: frintx
define double @test8(double %a) #0 {
entry:
  %call = tail call double @llvm.rint.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test9:
; CHECK: frintz
; CHECK-NOT: frintx
define float @test9(float %a) #0 {
entry:
  %call = tail call float @llvm.trunc.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test10:
; CHECK: frintz
; CHECK-NOT: frintx
define double @test10(double %a) #0 {
entry:
  %call = tail call double @llvm.trunc.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test11:
; CHECK: frinta
; CHECK-NOT: frintx
define float @test11(float %a) #0 {
entry:
  %call = tail call float @llvm.round.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test12:
; CHECK: frinta
; CHECK-NOT: frintx
define double @test12(double %a) #0 {
entry:
  %call = tail call double @llvm.round.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test13:
; CHECK-NOT: frintx
; CHECK: frintm
define float @test13(float %a) #1 {
entry:
  %call = tail call float @llvm.floor.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test14:
; CHECK-NOT: frintx
; CHECK: frintm
define double @test14(double %a) #1 {
entry:
  %call = tail call double @llvm.floor.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test15:
; CHECK-NOT: frintx
; CHECK: frintp
define float @test15(float %a) #1 {
entry:
  %call = tail call float @llvm.ceil.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test16:
; CHECK-NOT: frintx
; CHECK: frintp
define double @test16(double %a) #1 {
entry:
  %call = tail call double @llvm.ceil.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test17:
; CHECK-NOT: frintx
; CHECK: frintz
define float @test17(float %a) #1 {
entry:
  %call = tail call float @llvm.trunc.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test18:
; CHECK-NOT: frintx
; CHECK: frintz
define double @test18(double %a) #1 {
entry:
  %call = tail call double @llvm.trunc.f64(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: test19:
; CHECK-NOT: frintx
; CHECK: frinta
define float @test19(float %a) #1 {
entry:
  %call = tail call float @llvm.round.f32(float %a) nounwind readnone
  ret float %call
}

; CHECK-LABEL: test20:
; CHECK-NOT: frintx
; CHECK: frinta
define double @test20(double %a) #1 {
entry:
  %call = tail call double @llvm.round.f64(double %a) nounwind readnone
  ret double %call
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
