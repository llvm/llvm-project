; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2q | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define <4 x double> @foo_fmf(<4 x double> %a, <4 x double> %b) nounwind {
; CHECK-LABEL: @foo_fmf
; CHECK: qvfrsqrte
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfmsub
; CHECK-DAG: qvfnmsub
; CHECK: qvfmul
; CHECK: qvfmul
; CHECK: qvfnmsub
; CHECK: qvfmul
; CHECK: qvfmul
; CHECK: blr
entry:
  %x = call fast <4 x double> @llvm.sqrt.v4f64(<4 x double> %b)
  %r = fdiv fast <4 x double> %a, %x
  ret <4 x double> %r
}

define <4 x double> @foo_safe(<4 x double> %a, <4 x double> %b) nounwind {
; CHECK-LABEL: @foo_safe
; CHECK: fsqrt
; CHECK: fdiv
; CHECK: blr
entry:
  %x = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %b)
  %r = fdiv <4 x double> %a, %x
  ret <4 x double> %r
}

define <4 x double> @foof_fmf(<4 x double> %a, <4 x float> %b) nounwind {
; CHECK-LABEL: @foof_fmf
; CHECK: qvfrsqrtes
; CHECK-DAG: qvfmuls
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadd instead of a qvfnmsubs
; CHECK-DAG: qvfmadds
; CHECK-DAG: qvfmadds
; CHECK: qvfmuls
; CHECK: qvfmul
; CHECK: blr
entry:
  %x = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %y = fpext <4 x float> %x to <4 x double>
  %r = fdiv fast <4 x double> %a, %y
  ret <4 x double> %r
}

define <4 x double> @foof_safe(<4 x double> %a, <4 x float> %b) nounwind {
; CHECK-LABEL: @foof_safe
; CHECK: fsqrts
; CHECK: fdiv
; CHECK: blr
entry:
  %x = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %y = fpext <4 x float> %x to <4 x double>
  %r = fdiv <4 x double> %a, %y
  ret <4 x double> %r
}

define <4 x float> @food_fmf(<4 x float> %a, <4 x double> %b) nounwind {
; CHECK-LABEL: @food_fmf
; CHECK: qvfrsqrte
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfmsub
; CHECK-DAG: qvfnmsub
; CHECK: qvfmul
; CHECK: qvfmul
; CHECK: qvfnmsub
; CHECK: qvfmul
; CHECK: qvfrsp
; CHECK: qvfmuls
; CHECK: blr
entry:
  %x = call fast <4 x double> @llvm.sqrt.v4f64(<4 x double> %b)
  %y = fptrunc <4 x double> %x to <4 x float>
  %r = fdiv fast <4 x float> %a, %y
  ret <4 x float> %r
}

define <4 x float> @food_safe(<4 x float> %a, <4 x double> %b) nounwind {
; CHECK-LABEL: @food_safe
; CHECK: fsqrt
; CHECK: fdivs
; CHECK: blr
entry:
  %x = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %b)
  %y = fptrunc <4 x double> %x to <4 x float>
  %r = fdiv <4 x float> %a, %y
  ret <4 x float> %r
}

define <4 x float> @goo_fmf(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK-LABEL: @goo_fmf
; CHECK: qvfrsqrtes
; CHECK-DAG: qvfmuls
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadd instead of a qvfnmsubs
; CHECK-DAG: qvfmadds
; CHECK-DAG: qvfmadds
; CHECK: qvfmuls
; CHECK: qvfmuls
; CHECK: blr
entry:
  %x = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %r = fdiv fast <4 x float> %a, %x
  ret <4 x float> %r
}

define <4 x float> @goo_safe(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK-LABEL: @goo_safe
; CHECK: fsqrts
; CHECK: fdivs
; CHECK: blr
entry:
  %x = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %r = fdiv <4 x float> %a, %x
  ret <4 x float> %r
}

define <4 x double> @foo2_fmf(<4 x double> %a, <4 x double> %b) nounwind {
; CHECK-LABEL: @foo2_fmf
; CHECK: qvfre
; CHECK: qvfnmsub
; CHECK: qvfmadd
; CHECK: qvfnmsub
; CHECK: qvfmadd
; CHECK: qvfmul
; CHECK: blr
entry:
  %r = fdiv fast <4 x double> %a, %b
  ret <4 x double> %r
}

define <4 x double> @foo2_safe(<4 x double> %a, <4 x double> %b) nounwind {
; CHECK-LABEL: @foo2_safe
; CHECK: fdiv
; CHECK: blr
  %r = fdiv <4 x double> %a, %b
  ret <4 x double> %r
}

define <4 x float> @goo2_fmf(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK-LABEL: @goo2_fmf
; CHECK: qvfres
; CHECK: qvfnmsubs
; CHECK: qvfmadds
; CHECK: qvfmuls
; CHECK: blr
entry:
  %r = fdiv fast <4 x float> %a, %b
  ret <4 x float> %r
}

define <4 x float> @goo2_safe(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK-LABEL: @goo2_safe
; CHECK: fdivs
; CHECK: blr
entry:
  %r = fdiv <4 x float> %a, %b
  ret <4 x float> %r
}

define <4 x double> @foo3_fmf(<4 x double> %a) nounwind {
; CHECK-LABEL: @foo3_fmf
; CHECK: qvfrsqrte
; CHECK: qvfmul
; CHECK-DAG: qvfmsub
; CHECK-DAG: qvfcmpeq
; CHECK-DAG: qvfnmsub
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfnmsub
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfmul
; CHECK: qvfsel
; CHECK: blr
entry:
  %r = call fast <4 x double> @llvm.sqrt.v4f64(<4 x double> %a)
  ret <4 x double> %r
}

define <4 x double> @foo3_safe(<4 x double> %a) nounwind {
; CHECK-LABEL: @foo3_safe
; CHECK: fsqrt
; CHECK: blr
entry:
  %r = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %a)
  ret <4 x double> %r
}

define <4 x float> @goo3_fmf(<4 x float> %a) nounwind {
; CHECK-LABEL: @goo3_fmf
; CHECK: qvfrsqrtes
; CHECK: qvfmuls
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadds instead of a qvfnmsubs
; CHECK-DAG: qvfmadds
; CHECK-DAG: qvfcmpeq
; CHECK-DAG: qvfmadds
; CHECK-DAG: qvfmuls
; CHECK-DAG: qvfmuls
; CHECK: qvfsel
; CHECK: blr
entry:
  %r = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %r
}

define <4 x float> @goo3_safe(<4 x float> %a) nounwind {
; CHECK-LABEL: @goo3_safe
; CHECK: fsqrts
; CHECK: blr
entry:
  %r = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %r
}

