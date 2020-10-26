; RUN: opt -vector-library=SVML       -inject-tli-mappings        -S < %s | FileCheck %s  --check-prefixes=COMMON,SVML
; RUN: opt -vector-library=SVML       -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,SVML
; RUN: opt -vector-library=MASSV      -inject-tli-mappings        -S < %s | FileCheck %s  --check-prefixes=COMMON,MASSV
; RUN: opt -vector-library=MASSV      -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,MASSV
; RUN: opt -vector-library=Accelerate -inject-tli-mappings        -S < %s | FileCheck %s  --check-prefixes=COMMON,ACCELERATE
; RUN: opt -vector-library=LIBMVEC-X86 -inject-tli-mappings        -S < %s | FileCheck %s  --check-prefixes=COMMON,LIBMVEC-X86
; RUN: opt -vector-library=LIBMVEC-X86 -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,LIBMVEC-X86
; RUN: opt -vector-library=Accelerate -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,ACCELERATE

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; COMMON-LABEL: @llvm.compiler.used = appending global
; SVML-SAME:        [6 x i8*] [
; SVML-SAME:          i8* bitcast (<2 x double> (<2 x double>)* @__svml_sin2 to i8*),
; SVML-SAME:          i8* bitcast (<4 x double> (<4 x double>)* @__svml_sin4 to i8*),
; SVML-SAME:          i8* bitcast (<8 x double> (<8 x double>)* @__svml_sin8 to i8*),
; SVML-SAME:          i8* bitcast (<4 x float> (<4 x float>)* @__svml_log10f4 to i8*),
; SVML-SAME:          i8* bitcast (<8 x float> (<8 x float>)* @__svml_log10f8 to i8*),
; SVML-SAME:          i8* bitcast (<16 x float> (<16 x float>)* @__svml_log10f16 to i8*)
; MASSV-SAME:       [2 x i8*] [
; MASSV-SAME:         i8* bitcast (<2 x double> (<2 x double>)* @__sind2_massv to i8*),
; MASSV-SAME:         i8* bitcast (<4 x float> (<4 x float>)* @__log10f4_massv to i8*)
; ACCELERATE-SAME:  [1 x i8*] [
; ACCELERATE-SAME:    i8* bitcast (<4 x float> (<4 x float>)* @vlog10f to i8*)
; LIBMVEC-X86-SAME: [2 x i8*] [
; LIBMVEC-X86-SAME:   i8* bitcast (<2 x double> (<2 x double>)* @_ZGVbN2v_sin to i8*),
; LIBMVEC-X86-SAME:   i8* bitcast (<4 x double> (<4 x double>)* @_ZGVdN4v_sin to i8*)
; COMMON-SAME:      ], section "llvm.metadata"

define double @sin_f64(double %in) {
; COMMON-LABEL: @sin_f64(
; SVML:         call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; MASSV:        call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; ACCELERATE:   call double @sin(double %{{.*}})
; LIBMVEC-X86:  call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; No mapping of "sin" to a vector function for Accelerate.
; ACCELERATE-NOT: _ZGV_LLVM_{{.*}}_sin({{.*}})
  %call = tail call double @sin(double %in)
  ret double %call
}

declare double @sin(double) #0

define float @call_llvm.log10.f32(float %in) {
; COMMON-LABEL: @call_llvm.log10.f32(
; SVML:         call float @llvm.log10.f32(float %{{.*}})
; LIBMVEC-X86:      call float @llvm.log10.f32(float %{{.*}})
; MASSV:        call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; ACCELERATE:   call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; No mapping of "llvm.log10.f32" to a vector function for SVML.
; SVML-NOT:     _ZGV_LLVM_{{.*}}_llvm.log10.f32({{.*}})
; LIBMVEC-X86-NOT: _ZGV_LLVM_{{.*}}_llvm.log10.f32({{.*}})
  %call = tail call float @llvm.log10.f32(float %in)
  ret float %call
}

declare float @llvm.log10.f32(float) #0
attributes #0 = { nounwind readnone }

; SVML:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; SVML-SAME:   "_ZGV_LLVM_N2v_sin(__svml_sin2),
; SVML-SAME:   _ZGV_LLVM_N4v_sin(__svml_sin4),
; SVML-SAME:   _ZGV_LLVM_N8v_sin(__svml_sin8)" }

; MASSV:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; MASSV-SAME:   "_ZGV_LLVM_N2v_sin(__sind2_massv)" }
; MASSV:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; MASSV-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(__log10f4_massv)" }

; ACCELERATE:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; ACCELERATE-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(vlog10f)" }

; LIBMVEC-X86:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; LIBMVEC-X86-SAME:   "_ZGV_LLVM_N2v_sin(_ZGVbN2v_sin),
; LIBMVEC-X86-SAME:   _ZGV_LLVM_N4v_sin(_ZGVdN4v_sin)" }
