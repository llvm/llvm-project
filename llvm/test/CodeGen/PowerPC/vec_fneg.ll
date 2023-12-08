; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 | FileCheck %s -check-prefix=CHECK-NOVSX
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:          -mattr=+altivec -mattr=+vsx |  FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:          -mattr=+altivec -mattr=-vsx |  FileCheck %s \
; RUN:          -check-prefix=CHECK-NOVSX

define void @test_float(ptr %A) {
; CHECK-LABEL: test_float
; CHECK-NOVSX-LABEL: test_float
	%tmp2 = load <4 x float>, ptr %A
	%tmp3 = fsub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %tmp2
	store <4 x float> %tmp3, ptr %A
	ret void

; CHECK: xvnegsp
; CHECK: blr
; CHECK-NOVSX: vsubfp
; CHECK-NOVSX: blr

}

define void @test_double(ptr %A) {
; CHECK-LABEL: test_double
; CHECK-NOVSX-LABEL: test_double
	%tmp2 = load <2 x double>, ptr %A
	%tmp3 = fsub <2 x double> < double -0.000000e+00, double -0.000000e+00 >, %tmp2
	store <2 x double> %tmp3, ptr %A
	ret void

; CHECK: xvnegdp
; CHECK: blr
; CHECK-NOVSX: fneg
; CHECK-NOVSX: fneg
; CHECK-NOVSX: blr

}
