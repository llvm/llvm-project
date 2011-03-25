; RUN: llc < %s -mtriple=x86_64-linux -mattr=+64bit,+sse3 -print-failed-fuse-candidates |& FileCheck %s
; CHECK: fail
; CHECK-NOT: fail

declare float @test_f(float %f)
declare double @test_d(double %f)
declare <4 x float> @test_vf(<4 x float> %f)
declare <2 x double> @test_vd(<2 x double> %f)
declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>)
declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>)
declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>)
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8)
declare <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>)
declare <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float>, <4 x float>)
declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>)
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double>, <2 x double>, i8)
declare <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double>, <2 x double>)

define float @foo(float %f) {
  %a = call float @test_f(float %f)
  %t = call float @llvm.sqrt.f32(float %f)
  ret float %t
}
define double @doo(double %f) {
  %a = call double @test_d(double %f)
  %t = call double @llvm.sqrt.f64(double %f)
  ret double %t
}
define <4 x float> @a0(<4 x float> %f) {
  %a = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %f)
  ret <4 x float> %t
}
define <4 x float> @a1(<4 x float> %f) {
  %a = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %f)
  ret <4 x float> %t
}
define <4 x float> @a2(<4 x float> %f) {
  %a = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %f)
  ret <4 x float> %t
}
define <4 x float> @b3(<4 x float> %f) {
  %y = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %y, <4 x float> %f)
  ret <4 x float> %t
}
define <4 x float> @b4(<4 x float> %f) {
  %y = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %y, <4 x float> %f)
  ret <4 x float> %t
}
define <4 x float> @b5(<4 x float> %f) {
  %y = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %y, <4 x float> %f, i8 7)
  ret <4 x float> %t
}
define <4 x float> @b6(<4 x float> %f) {
  %y = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float> %y, <4 x float> %f)
  ret <4 x float> %t
}
define <4 x float> @b7(<4 x float> %f) {
  %y = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %y, <4 x float> %f)
  ret <4 x float> %t
}
define <4 x float> @b8(<4 x float> %f) {
  %y = call <4 x float> @test_vf(<4 x float> %f)
  %t = call <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float> %y, <4 x float> %f)
  ret <4 x float> %t
}
define <2 x double> @c1(<2 x double> %f) {
  %a = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double> %f)
  ret <2 x double> %t
}
define <2 x double> @d3(<2 x double> %f) {
  %y = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %y, <2 x double> %f)
  ret <2 x double> %t
}
define <2 x double> @d4(<2 x double> %f) {
  %y = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %y, <2 x double> %f)
  ret <2 x double> %t
}
define <2 x double> @d5(<2 x double> %f) {
  %y = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %y, <2 x double> %f, i8 7)
  ret <2 x double> %t
}
define <2 x double> @d6(<2 x double> %f) {
  %y = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double> %y, <2 x double> %f)
  ret <2 x double> %t
}
define <2 x double> @d7(<2 x double> %f) {
  %y = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double> %y, <2 x double> %f)
  ret <2 x double> %t
}
define <2 x double> @d8(<2 x double> %f) {
  %y = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double> %y, <2 x double> %f)
  ret <2 x double> %t
}

; This one should fail to fuse.
define <2 x double> @z0(<2 x double> %f) {
  %y = call <2 x double> @test_vd(<2 x double> %f)
  %t = call <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double> %f, <2 x double> %y)
  ret <2 x double> %t
}
