; RUN: llc < %s -mcpu=sm_60 -verify-machineinstrs | FileCheck %s --implicit-check-not=call
; RUN: llc < %s -mcpu=sm_90 -mattr=+ptx78 -verify-machineinstrs | FileCheck %s --implicit-check-not=call
; RUN: llc < %s -mcpu=sm_60 -O0 -verify-machineinstrs | FileCheck %s --implicit-check-not=call
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_60 -verify-machineinstrs | FileCheck %s --implicit-check-not=call
; RUN: llc < %s -mcpu=sm_60 -enable-new-pm -verify-machineinstrs | FileCheck %s --implicit-check-not=call
; RUN: llc < %s -mcpu=sm_60 -enable-new-pm -O0 -verify-machineinstrs | FileCheck %s --implicit-check-not=call

target triple = "nvptx64-unknown-cuda"

; Precise frem is expanded inline, including for promoted types and vectors.
; Every configuration must avoid introducing runtime library calls.

define half @frem_f16(half %a, half %b) {
; CHECK-LABEL: frem_f16(
; CHECK-DAG: fma.rn.f32
; CHECK-DAG: ret;
  %r = frem half %a, %b
  ret half %r
}

define bfloat @frem_bf16(bfloat %a, bfloat %b) {
; CHECK-LABEL: frem_bf16(
; CHECK-DAG: fma.rn.f32
; CHECK-DAG: ret;
  %r = frem bfloat %a, %b
  ret bfloat %r
}

define float @frem_f32(float %a, float %b) {
; CHECK-LABEL: frem_f32(
; CHECK-DAG: fma.rn.f32
; CHECK-DAG: ret;
  %r = frem float %a, %b
  ret float %r
}

define double @frem_f64(double %a, double %b) {
; CHECK-LABEL: frem_f64(
; CHECK-DAG: fma.rn.f64
; CHECK-DAG: ret;
  %r = frem double %a, %b
  ret double %r
}

define <2 x half> @frem_v2f16(<2 x half> %a, <2 x half> %b) {
; CHECK-LABEL: frem_v2f16(
; CHECK-DAG: fma.rn.f32
; CHECK-DAG: ret;
  %r = frem <2 x half> %a, %b
  ret <2 x half> %r
}

define <2 x bfloat> @frem_v2bf16(<2 x bfloat> %a, <2 x bfloat> %b) {
; CHECK-LABEL: frem_v2bf16(
; CHECK-DAG: fma.rn.f32
; CHECK-DAG: ret;
  %r = frem <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

define <2 x float> @frem_v2f32(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: frem_v2f32(
; CHECK-DAG: fma.rn.f32
; CHECK-DAG: ret;
  %r = frem <2 x float> %a, %b
  ret <2 x float> %r
}

define <2 x double> @frem_v2f64(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: frem_v2f64(
; CHECK-DAG: fma.rn.f64
; CHECK-DAG: ret;
  %r = frem <2 x double> %a, %b
  ret <2 x double> %r
}

; Other fast-math flags still use the precise expansion without afn.
define float @frem_other_flags(float %a, float %b) {
; CHECK-LABEL: frem_other_flags(
; CHECK-DAG: fma.rn.f32
; CHECK-DAG: ret;
  %r = frem reassoc nnan ninf nsz arcp contract float %a, %b
  ret float %r
}
