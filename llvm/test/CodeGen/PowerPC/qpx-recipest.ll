; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2q -enable-unsafe-fp-math | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2q | FileCheck -check-prefix=CHECK-SAFE %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define <4 x double> @foo(<4 x double> %a, <4 x double> %b) nounwind {
entry:
  %x = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %b)
  %r = fdiv <4 x double> %a, %x
  ret <4 x double> %r

; CHECK-LABEL: @foo
; CHECK: qvfrsqrte
; CHECK-DAG: qvfmul
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadd instead of a qvfnmsub
; CHECK-DAG: qvfmadd
; CHECK-DAG: qvfmadd
; CHECK: qvfmul
; CHECK: qvfmul
; CHECK: qvfmadd
; CHECK: qvfmul
; CHECK: qvfmul
; CHECK: blr

; CHECK-SAFE-LABEL: @foo
; CHECK-SAFE: fsqrt
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define <4 x double> @foof(<4 x double> %a, <4 x float> %b) nounwind {
entry:
  %x = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %y = fpext <4 x float> %x to <4 x double>
  %r = fdiv <4 x double> %a, %y
  ret <4 x double> %r

; CHECK-LABEL: @foof
; CHECK: qvfrsqrtes
; CHECK-DAG: qvfmuls
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadd instead of a qvfnmsubs
; CHECK-DAG: qvfmadds
; CHECK-DAG: qvfmadds
; CHECK: qvfmuls
; CHECK: qvfmul
; CHECK: blr

; CHECK-SAFE-LABEL: @foof
; CHECK-SAFE: fsqrts
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define <4 x float> @food(<4 x float> %a, <4 x double> %b) nounwind {
entry:
  %x = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %b)
  %y = fptrunc <4 x double> %x to <4 x float>
  %r = fdiv <4 x float> %a, %y
  ret <4 x float> %r

; CHECK-LABEL: @food
; CHECK: qvfrsqrte
; CHECK-DAG: qvfmul
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadd instead of a qvfnmsub
; CHECK-DAG: qvfmadd
; CHECK-DAG: qvfmadd
; CHECK: qvfmul
; CHECK: qvfmul
; CHECK: qvfmadd
; CHECK: qvfmul
; CHECK: qvfrsp
; CHECK: qvfmuls
; CHECK: blr

; CHECK-SAFE-LABEL: @food
; CHECK-SAFE: fsqrt
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define <4 x float> @goo(<4 x float> %a, <4 x float> %b) nounwind {
entry:
  %x = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %r = fdiv <4 x float> %a, %x
  ret <4 x float> %r

; CHECK-LABEL: @goo
; CHECK: qvfrsqrtes
; CHECK-DAG: qvfmuls
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadd instead of a qvfnmsubs
; CHECK-DAG: qvfmadds
; CHECK-DAG: qvfmadds
; CHECK: qvfmuls
; CHECK: qvfmuls
; CHECK: blr

; CHECK-SAFE-LABEL: @goo
; CHECK-SAFE: fsqrts
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define <4 x double> @foo2(<4 x double> %a, <4 x double> %b) nounwind {
entry:
  %r = fdiv <4 x double> %a, %b
  ret <4 x double> %r

; CHECK-LABEL: @foo2
; CHECK: qvfre
; CHECK: qvfnmsub
; CHECK: qvfmadd
; CHECK: qvfnmsub
; CHECK: qvfmadd
; CHECK: qvfmul
; CHECK: blr

; CHECK-SAFE-LABEL: @foo2
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define <4 x float> @goo2(<4 x float> %a, <4 x float> %b) nounwind {
entry:
  %r = fdiv <4 x float> %a, %b
  ret <4 x float> %r

; CHECK-LABEL: @goo2
; CHECK: qvfres
; CHECK: qvfnmsubs
; CHECK: qvfmadds
; CHECK: qvfmuls
; CHECK: blr

; CHECK-SAFE-LABEL: @goo2
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define <4 x double> @foo3(<4 x double> %a) nounwind {
entry:
  %r = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %a)
  ret <4 x double> %r

; CHECK-LABEL: @foo3
; CHECK: qvfrsqrte
; CHECK: qvfmul
; FIXME: We're currently loading two constants here (1.5 and -1.5), and using
;        an qvfmadd instead of a qvfnmsub
; CHECK-DAG: qvfmadd
; CHECK-DAG: qvfcmpeq
; CHECK-DAG: qvfmadd
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfmadd
; CHECK-DAG: qvfmul
; CHECK-DAG: qvfmul
; CHECK: qvfsel
; CHECK: blr

; CHECK-SAFE-LABEL: @foo3
; CHECK-SAFE: fsqrt
; CHECK-SAFE: blr
}

define <4 x float> @goo3(<4 x float> %a) nounwind {
entry:
  %r = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %r

; CHECK-LABEL: @goo3
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

; CHECK-SAFE-LABEL: @goo3
; CHECK-SAFE: fsqrts
; CHECK-SAFE: blr
}

