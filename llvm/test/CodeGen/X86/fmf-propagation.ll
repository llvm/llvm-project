; REQUIRES: asserts
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx512f -debug-only=isel < %s -o /dev/null 2>&1 | FileCheck %s

; This tests the propagation of fast-math-flags from IR instructions to SDNodeFlags.

; CHECK-LABEL: Initial selection DAG: %bb.0 'fmf_transfer:'

; CHECK:         t5: f32 = fadd nsz t2, t4
; CHECK-NEXT:    t6: f32 = fadd arcp t5, t4
; CHECK-NEXT:    t7: f32 = fadd nnan t6, t4
; CHECK-NEXT:    t8: f32 = fadd ninf t7, t4
; CHECK-NEXT:    t9: f32 = fadd contract t8, t4
; CHECK-NEXT:    t10: f32 = fadd afn t9, t4
; CHECK-NEXT:    t11: f32 = fadd reassoc t10, t4
; CHECK-NEXT:    t12: f32 = fadd nnan ninf nsz arcp contract afn reassoc t11, t4

; CHECK: Optimized lowered selection DAG: %bb.0 'fmf_transfer:'

define float @fmf_transfer(float %x, float %y) {
  %f1 = fadd nsz float %x, %y
  %f2 = fadd arcp float %f1, %y
  %f3 = fadd nnan float %f2, %y
  %f4 = fadd ninf float %f3, %y
  %f5 = fadd contract float %f4, %y
  %f6 = fadd afn float %f5, %y
  %f7 = fadd reassoc float %f6, %y
  %f8 = fadd fast float %f7, %y
  ret float %f8
}

; CHECK-LABEL: Optimized type-legalized selection DAG: %bb.0 'fmf_setcc:'
; CHECK: t13: i8 = setcc nnan ninf nsz arcp contract afn reassoc t2, ConstantFP:f32<0.000000e+00>, setlt:ch

define float @fmf_setcc(float %x, float %y) {
  %cmp = fcmp fast ult float %x, 0.0
  %ret = select i1 %cmp, float %x, float %y
  ret float %ret
}

; CHECK-LABEL: Initial selection DAG: %bb.0 'fmf_setcc_canon:'
; CHECK: t14: i8 = setcc nnan ninf nsz arcp contract afn reassoc t2, ConstantFP:f32<0.000000e+00>, setgt:ch
define float @fmf_setcc_canon(float %x, float %y) {
  %cmp = fcmp fast ult float 0.0, %x
  %ret = select i1 %cmp, float %x, float %y
  ret float %ret
}

declare <16 x float> @llvm.x86.avx512.vfmadd.ps.512(<16 x float>, <16 x float>, <16 x float>, i32)

; Check that FMF are propagated twice: from IR to x86-specific node and from x86-specific node to generic node.

; CHECK-LABEL: Initial selection DAG: %bb.0 'fmf_target_intrinsic:'
; CHECK:       v16f32 = llvm.x86.avx512.vfmadd.ps.512 ninf nsz TargetConstant:i64<{{.*}}>
; CHECK:       v16f32 = llvm.x86.avx512.vfmadd.ps.512 nsz TargetConstant:i64<{{.*}}>

; CHECK-LABEL: Legalized selection DAG: %bb.0 'fmf_target_intrinsic:'
; CHECK:       v16f32 = fma ninf nsz t{{.*}}
; CHECK:       v16f32 = fma nsz t{{.*}}

define <16 x float> @fmf_target_intrinsic(<16 x float> %a, <16 x float> %b, <16 x float> %c) nounwind {
  %t0 = tail call ninf nsz <16 x float> @llvm.x86.avx512.vfmadd.ps.512(<16 x float> %a, <16 x float> %b, <16 x float> %c, i32 4)
  %t1 = tail call nsz <16 x float> @llvm.x86.avx512.vfmadd.ps.512(<16 x float> %t0, <16 x float> %b, <16 x float> %c, i32 4)
  ret <16 x float> %t1
}
