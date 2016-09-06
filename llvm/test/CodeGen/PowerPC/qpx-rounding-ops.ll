; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2q | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2q -enable-unsafe-fp-math | FileCheck -check-prefix=CHECK-FM %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <4 x float> @test1(<4 x float> %x) nounwind  {
  %call = tail call <4 x float> @llvm.floor.v4f32(<4 x float> %x) nounwind readnone
  ret <4 x float> %call

; CHECK: test1:
; CHECK: qvfrim 1, 1

; CHECK-FM: test1:
; CHECK-FM: qvfrim 1, 1
}

declare <4 x float> @llvm.floor.v4f32(<4 x float>) nounwind readnone

define <4 x double> @test2(<4 x double> %x) nounwind  {
  %call = tail call <4 x double> @llvm.floor.v4f64(<4 x double> %x) nounwind readnone
  ret <4 x double> %call

; CHECK: test2:
; CHECK: qvfrim 1, 1

; CHECK-FM: test2:
; CHECK-FM: qvfrim 1, 1
}

declare <4 x double> @llvm.floor.v4f64(<4 x double>) nounwind readnone

define <4 x float> @test3(<4 x float> %x) nounwind  {
  %call = tail call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %x) nounwind readnone
  ret <4 x float> %call

; CHECK: test3:
; CHECK-NOT: qvfrin

; CHECK-FM: test3:
; CHECK-FM-NOT: qvfrin
}

declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>) nounwind readnone

define <4 x double> @test4(<4 x double> %x) nounwind  {
  %call = tail call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %x) nounwind readnone
  ret <4 x double> %call

; CHECK: test4:
; CHECK-NOT: qvfrin

; CHECK-FM: test4:
; CHECK-FM-NOT: qvfrin
}

declare <4 x double> @llvm.nearbyint.v4f64(<4 x double>) nounwind readnone

define <4 x float> @test5(<4 x float> %x) nounwind  {
  %call = tail call <4 x float> @llvm.ceil.v4f32(<4 x float> %x) nounwind readnone
  ret <4 x float> %call

; CHECK: test5:
; CHECK: qvfrip 1, 1

; CHECK-FM: test5:
; CHECK-FM: qvfrip 1, 1
}

declare <4 x float> @llvm.ceil.v4f32(<4 x float>) nounwind readnone

define <4 x double> @test6(<4 x double> %x) nounwind  {
  %call = tail call <4 x double> @llvm.ceil.v4f64(<4 x double> %x) nounwind readnone
  ret <4 x double> %call

; CHECK: test6:
; CHECK: qvfrip 1, 1

; CHECK-FM: test6:
; CHECK-FM: qvfrip 1, 1
}

declare <4 x double> @llvm.ceil.v4f64(<4 x double>) nounwind readnone

define <4 x float> @test9(<4 x float> %x) nounwind  {
  %call = tail call <4 x float> @llvm.trunc.v4f32(<4 x float> %x) nounwind readnone
  ret <4 x float> %call

; CHECK: test9:
; CHECK: qvfriz 1, 1

; CHECK-FM: test9:
; CHECK-FM: qvfriz 1, 1
}

declare <4 x float> @llvm.trunc.v4f32(<4 x float>) nounwind readnone

define <4 x double> @test10(<4 x double> %x) nounwind  {
  %call = tail call <4 x double> @llvm.trunc.v4f64(<4 x double> %x) nounwind readnone
  ret <4 x double> %call

; CHECK: test10:
; CHECK: qvfriz 1, 1

; CHECK-FM: test10:
; CHECK-FM: qvfriz 1, 1
}

declare <4 x double> @llvm.trunc.v4f64(<4 x double>) nounwind readnone

