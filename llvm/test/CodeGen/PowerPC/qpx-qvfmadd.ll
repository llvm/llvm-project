; RUN: llc -verify-machineinstrs -stop-after=finalize-isel < %s -mcpu=a2q | FileCheck %s
target triple = "powerpc64-bgq-linux"

define <2 x double> @test_qvfmadd(<2 x double> %0, <2 x double> %1, <2 x double> %2) {
; CHECK: test_qvfmadd
; CHECK: QVFMADD %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <2 x double> %2, %1
  %5 = fadd reassoc nsz <2 x double> %4, %0
  ret <2 x double> %5
}

define <4 x float> @test_qvfmadds(<4 x float> %0, <4 x float> %1, <4 x float> %2) {
; CHECK: test_qvfmadds
; CHECK: QVFMADDSs %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <4 x float> %2, %1
  %5 = fadd reassoc nsz <4 x float> %4, %0
  ret <4 x float> %5
}

define <2 x double> @test_qvfnmadd(<2 x double> %0, <2 x double> %1, <2 x double> %2) {
; CHECK: test_qvfnmadd
; CHECK: QVFNMADD %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <2 x double> %2, %1
  %5 = fadd reassoc nsz <2 x double> %4, %0
  %6 = fneg reassoc nsz <2 x double> %5
  ret <2 x double> %6
}

define <4 x float> @test_qvfnmadds(<4 x float> %0, <4 x float> %1, <4 x float> %2) {
; CHECK: test_qvfnmadds
; CHECK: QVFNMADDSs %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <4 x float> %2, %1
  %5 = fadd reassoc nsz <4 x float> %4, %0
  %6 = fneg reassoc nsz <4 x float> %5
  ret <4 x float> %6
}

define <2 x double> @test_qvfmsub(<2 x double> %0, <2 x double> %1, <2 x double> %2) {
; CHECK: test_qvfmsub
; CHECK: QVFMSUB %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <2 x double> %2, %1
  %5 = fsub reassoc nsz <2 x double> %4, %0
  ret <2 x double> %5
}

define <4 x float> @test_qvfmsubs(<4 x float> %0, <4 x float> %1, <4 x float> %2) {
; CHECK: test_qvfmsubs
; CHECK: QVFMSUBSs %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <4 x float> %2, %1
  %5 = fsub reassoc nsz <4 x float> %4, %0
  ret <4 x float> %5
}

define <2 x double> @test_qvfnmsub(<2 x double> %0, <2 x double> %1, <2 x double> %2) {
; CHECK: test_qvfnmsub
; CHECK: QVFNMSUB %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <2 x double> %2, %1
  %5 = fsub reassoc nsz <2 x double> %4, %0
  %6 = fneg reassoc nsz <2 x double> %5
  ret <2 x double> %6
}

define <4 x float> @test_qvfnmsubs(<4 x float> %0, <4 x float> %1, <4 x float> %2) {
; CHECK: test_qvfnmsubs
; CHECK: QVFNMSUBSs %2, %1, %0, implicit $rm
;
  %4 = fmul reassoc nsz <4 x float> %2, %1
  %5 = fsub reassoc nsz <4 x float> %4, %0
  %6 = fneg reassoc nsz <4 x float> %5
  ret <4 x float> %6
}

