; RUN: split-file %s %t
; RUN: llc --filetype=obj %t/minimum.ll -o - | dxil-dis -o - | FileCheck %s --check-prefix=LEGACY
; RUN: llc --filetype=obj %t/minimum.ll --stop-after=dxil-writer -o - | dxil-dis -o - | FileCheck %s --check-prefix=LEGACY
; RUN: llc --filetype=obj %t/native.ll -o - | dxil-dis -o - | FileCheck %s --check-prefix=MODERN
; RUN: llc --filetype=obj %t/native.ll --stop-after=dxil-writer -o - | dxil-dis -o - | FileCheck %s --check-prefix=MODERN
; RUN: llc --filetype=obj %t/no-low-precision.ll -o - | dxil-dis -o - | FileCheck %s --check-prefix=LEGACY
; RUN: llc --filetype=obj %t/no-low-precision.ll --stop-after=dxil-writer -o - | dxil-dis -o - | FileCheck %s --check-prefix=LEGACY
; RUN: llc --filetype=obj %t/native-no-low-precision.ll -o - | dxil-dis -o - | FileCheck %s --check-prefix=MODERN
; RUN: llc --filetype=obj %t/native-no-low-precision.ll --stop-after=dxil-writer -o - | dxil-dis -o - | FileCheck %s --check-prefix=MODERN

; Legacy layout is the default, even without low-precision types. Native low
; precision selects the modern layout regardless of the types used.
; LEGACY: target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
; MODERN: target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"

;--- minimum.ll
target triple = "dxil-unknown-shadermodel6.7-library"

define half @add(half %a, half %b) "hlsl.export" {
  %sum = fadd half %a, %b
  ret half %sum
}

;--- native.ll
target triple = "dxil-unknown-shadermodel6.7-library"

define half @add(half %a, half %b) "hlsl.export" {
  %sum = fadd half %a, %b
  ret half %sum
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.nativelowprec", i32 1}

;--- no-low-precision.ll
target triple = "dxil-unknown-shadermodel6.7-library"

define float @add(float %a, float %b) "hlsl.export" {
  %sum = fadd float %a, %b
  ret float %sum
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.nativelowprec", i32 0}

;--- native-no-low-precision.ll
target triple = "dxil-unknown-shadermodel6.7-library"

define float @add(float %a, float %b) "hlsl.export" {
  %sum = fadd float %a, %b
  ret float %sum
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.nativelowprec", i32 1}
