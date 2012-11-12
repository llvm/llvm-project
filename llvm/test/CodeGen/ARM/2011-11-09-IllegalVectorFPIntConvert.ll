; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <2 x i32> @test1(<2 x double>* %A) {
; CHECK: test1
; CHECK: vcvt.s32.f64
; CHECK: vcvt.s32.f64
  %tmp1 = load <2 x double>* %A
	%tmp2 = fptosi <2 x double> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <2 x i32> @test2(<2 x double>* %A) {
; CHECK: test2
; CHECK: vcvt.u32.f64
; CHECK: vcvt.u32.f64
  %tmp1 = load <2 x double>* %A
	%tmp2 = fptoui <2 x double> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <2 x double> @test3(<2 x i32>* %A) {
; CHECK: test3
; CHECK: vcvt.f64.s32
; CHECK: vcvt.f64.s32
  %tmp1 = load <2 x i32>* %A
	%tmp2 = sitofp <2 x i32> %tmp1 to <2 x double>
	ret <2 x double> %tmp2
}

define <2 x double> @test4(<2 x i32>* %A) {
; CHECK: test4
; CHECK: vcvt.f64.u32
; CHECK: vcvt.f64.u32
  %tmp1 = load <2 x i32>* %A
	%tmp2 = uitofp <2 x i32> %tmp1 to <2 x double>
	ret <2 x double> %tmp2
}
