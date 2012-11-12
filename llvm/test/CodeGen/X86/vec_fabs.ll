; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=corei7-avx | FileCheck %s


define <2 x double> @fabs_v2f64(<2 x double> %p)
{
  ; CHECK: fabs_v2f64
  ; CHECK: vandps
  %t = call <2 x double> @llvm.fabs.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.fabs.v2f64(<2 x double> %p)

define <4 x float> @fabs_v4f32(<4 x float> %p)
{
  ; CHECK: fabs_v4f32
  ; CHECK: vandps
  %t = call <4 x float> @llvm.fabs.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.fabs.v4f32(<4 x float> %p)

define <4 x double> @fabs_v4f64(<4 x double> %p)
{
  ; CHECK: fabs_v4f64
  ; CHECK: vandps
  %t = call <4 x double> @llvm.fabs.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.fabs.v4f64(<4 x double> %p)

define <8 x float> @fabs_v8f32(<8 x float> %p)
{
  ; CHECK: fabs_v8f32
  ; CHECK: vandps
  %t = call <8 x float> @llvm.fabs.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.fabs.v8f32(<8 x float> %p)
