; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -march=ppc64le \
; RUN:          -mattr=+altivec -mattr=+vsx |  FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -march=ppc64le \
; RUN:          -mattr=+altivec -mattr=-vsx |  FileCheck %s \
; RUN:          -check-prefix=CHECK-NOVSX

define <4 x float> @test_float(<4 x float> %aa) #0 {

; CHECK-LABEL: test_float
; CHECK-NOVSX-LABEL: test_float
; CHECK-NOVSX-LABEL: test_float

  entry:
    %0 = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %aa) #2
    ret <4 x float> %0
}
; Function Attrs: nounwind readnone
declare <4 x float> @llvm.fabs.v4f32(<4 x float>) #1

; CHECK: xvabssp
; CHECK: blr
; CHECK-NOVSX: fabs
; CHECK-NOVSX: fabs
; CHECK-NOVSX: fabs
; CHECK-NOVSX: fabs
; CHECK-NOVSX: blr

define <4 x float> @test2_float(<4 x float> %aa) #0 {

; CHECK-LABEL: test2_float
; CHECK-NOVSX-LABEL: test2_float

  entry:
    %0 = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %aa) #2
    %sub = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00,
                             float -0.000000e+00, float -0.000000e+00>, %0
    ret <4 x float> %sub
}

; CHECK: xvnabssp
; CHECK: blr
; CHECK-NOVSX: vspltisb
; CHECK-NOVSX: fabs
; CHECK-NOVSX: fabs
; CHECK-NOVSX: fabs
; CHECK-NOVSX: fabs
; CHECK-NOVSX: vsubfp
; CHECK-NOVSX: blr

define <2 x double> @test_double(<2 x double> %aa) #0 {

; CHECK-LABEL: test_double
; CHECK-NOVSX-LABEL: test_double

  entry:
    %0 = tail call <2 x double> @llvm.fabs.v2f64(<2 x double> %aa) #2
    ret <2 x double> %0
}

; Function Attrs: nounwind readnone
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) #1

; CHECK: xvabsdp
; CHECK: blr
; CHECK-NOVSX: fabs
; CHECK-NOVSX: fabs
; CHECK-NOVSX: blr

define <2 x double> @foo(<2 x double> %aa) #0 {
  entry:
    %0 = tail call <2 x double> @llvm.fabs.v2f64(<2 x double> %aa) #2
    %sub = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %0
    ret <2 x double> %sub
}

; CHECK: xvnabsdp
; CHECK: blr
; CHECK-NOVSX: fnabs
; CHECK-NOVSX: fnabs
; CHECK-NOVSX: blr
