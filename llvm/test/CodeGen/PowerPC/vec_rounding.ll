; RUN: llc -verify-machineinstrs -mcpu=pwr6 -mattr=+altivec < %s | FileCheck %s

; Check vector round to single-precision toward -infinity (vrfim)
; instruction generation using Altivec.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <2 x double> @llvm.floor.v2f64(<2 x double> %p)
define <2 x double> @floor_v2f64(<2 x double> %p)
{
  %t = call <2 x double> @llvm.floor.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
; CHECK-LABEL: floor_v2f64:
; CHECK: frim
; CHECK: frim

declare <4 x double> @llvm.floor.v4f64(<4 x double> %p)
define <4 x double> @floor_v4f64(<4 x double> %p)
{
  %t = call <4 x double> @llvm.floor.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
; CHECK-LABEL: floor_v4f64:
; CHECK: frim
; CHECK: frim
; CHECK: frim
; CHECK: frim

declare <2 x double> @llvm.ceil.v2f64(<2 x double> %p)
define <2 x double> @ceil_v2f64(<2 x double> %p)
{
  %t = call <2 x double> @llvm.ceil.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
; CHECK-LABEL: ceil_v2f64:
; CHECK: frip
; CHECK: frip

declare <4 x double> @llvm.ceil.v4f64(<4 x double> %p)
define <4 x double> @ceil_v4f64(<4 x double> %p)
{
  %t = call <4 x double> @llvm.ceil.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
; CHECK-LABEL: ceil_v4f64:
; CHECK: frip
; CHECK: frip
; CHECK: frip
; CHECK: frip

declare <2 x double> @llvm.trunc.v2f64(<2 x double> %p)
define <2 x double> @trunc_v2f64(<2 x double> %p)
{
  %t = call <2 x double> @llvm.trunc.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
; CHECK-LABEL: trunc_v2f64:
; CHECK: friz
; CHECK: friz

declare <4 x double> @llvm.trunc.v4f64(<4 x double> %p)
define <4 x double> @trunc_v4f64(<4 x double> %p)
{
  %t = call <4 x double> @llvm.trunc.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
; CHECK-LABEL: trunc_v4f64:
; CHECK: friz
; CHECK: friz
; CHECK: friz
; CHECK: friz

declare <2 x double> @llvm.nearbyint.v2f64(<2 x double> %p)
define <2 x double> @nearbyint_v2f64(<2 x double> %p)
{
  %t = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
; CHECK-LABEL: nearbyint_v2f64:
; CHECK: bl nearbyint
; CHECK: bl nearbyint

declare <4 x double> @llvm.nearbyint.v4f64(<4 x double> %p)
define <4 x double> @nearbyint_v4f64(<4 x double> %p)
{
  %t = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
; CHECK-LABEL: nearbyint_v4f64:
; CHECK: bl nearbyint
; CHECK: bl nearbyint
; CHECK: bl nearbyint
; CHECK: bl nearbyint


declare <4 x float> @llvm.floor.v4f32(<4 x float> %p)
define <4 x float> @floor_v4f32(<4 x float> %p)
{
  %t = call <4 x float> @llvm.floor.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
; CHECK-LABEL: floor_v4f32:
; CHECK: vrfim

declare <8 x float> @llvm.floor.v8f32(<8 x float> %p)
define <8 x float> @floor_v8f32(<8 x float> %p)
{
  %t = call <8 x float> @llvm.floor.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
; CHECK-LABEL: floor_v8f32:
; CHECK: vrfim
; CHECK: vrfim

declare <4 x float> @llvm.ceil.v4f32(<4 x float> %p)
define <4 x float> @ceil_v4f32(<4 x float> %p)
{
  %t = call <4 x float> @llvm.ceil.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
; CHECK-LABEL: ceil_v4f32:
; CHECK: vrfip

declare <8 x float> @llvm.ceil.v8f32(<8 x float> %p)
define <8 x float> @ceil_v8f32(<8 x float> %p)
{
  %t = call <8 x float> @llvm.ceil.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
; CHECK-LABEL: ceil_v8f32:
; CHECK: vrfip
; CHECK: vrfip

declare <4 x float> @llvm.trunc.v4f32(<4 x float> %p)
define <4 x float> @trunc_v4f32(<4 x float> %p)
{
  %t = call <4 x float> @llvm.trunc.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
; CHECK-LABEL: trunc_v4f32:
; CHECK: vrfiz

declare <8 x float> @llvm.trunc.v8f32(<8 x float> %p)
define <8 x float> @trunc_v8f32(<8 x float> %p)
{
  %t = call <8 x float> @llvm.trunc.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
; CHECK-LABEL: trunc_v8f32:
; CHECK: vrfiz
; CHECK: vrfiz

declare <4 x float> @llvm.nearbyint.v4f32(<4 x float> %p)
define <4 x float> @nearbyint_v4f32(<4 x float> %p)
{
  %t = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
; CHECK-LABEL: nearbyint_v4f32:
; CHECK: vrfin

declare <8 x float> @llvm.nearbyint.v8f32(<8 x float> %p)
define <8 x float> @nearbyint_v8f32(<8 x float> %p)
{
  %t = call <8 x float> @llvm.nearbyint.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
; CHECK-LABEL: nearbyint_v8f32:
; CHECK: vrfin
; CHECK: vrfin
