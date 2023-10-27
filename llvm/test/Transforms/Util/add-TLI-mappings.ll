; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=SVML -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,SVML
; RUN: opt -mtriple=powerpc64-unknown-linux-gnu -vector-library=MASSV -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,MASSV
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=LIBMVEC-X86 -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,LIBMVEC-X86
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=Accelerate -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,ACCELERATE
; RUN: opt -mtriple=aarch64-unknown-linux-gnu -vector-library=sleefgnuabi -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,SLEEFGNUABI
; RUN: opt -mtriple=aarch64-unknown-linux-gnu -vector-library=ArmPL -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,ARMPL

; COMMON-LABEL: @llvm.compiler.used = appending global
; SVML-SAME:        [6 x ptr] [
; SVML-SAME:          ptr @__svml_sin2,
; SVML-SAME:          ptr @__svml_sin4,
; SVML-SAME:          ptr @__svml_sin8,
; SVML-SAME:          ptr @__svml_log10f4,
; SVML-SAME:          ptr @__svml_log10f8,
; SVML-SAME:          ptr @__svml_log10f16
; MASSV-SAME:       [2 x ptr] [
; MASSV-SAME:         ptr @__sind2,
; MASSV-SAME:         ptr @__log10f4
; ACCELERATE-SAME:  [1 x ptr] [
; ACCELERATE-SAME:    ptr @vlog10f
; LIBMVEC-X86-SAME: [2 x ptr] [
; LIBMVEC-X86-SAME:   ptr @_ZGVbN2v_sin,
; LIBMVEC-X86-SAME:   ptr @_ZGVdN4v_sin
; SLEEFGNUABI-SAME: [4 x ptr] [
; SLEEFGNUABI-SAME:   ptr @_ZGVnN2v_sin,
; SLEEFGNUABI-SAME:   ptr @_ZGVsMxv_sin,
; SLEEFGNUABI_SAME;   ptr @_ZGVnN4v_log10f,
; SLEEFGNUABI-SAME:   ptr @_ZGVsMxv_log10f
; ARMPL-SAME:       [4 x ptr] [
; ARMPL-SAME:         ptr @armpl_vsinq_f64,
; ARMPL-SAME:         ptr @armpl_svsin_f64_x,
; ARMPL-SAME:         ptr @armpl_vlog10q_f32,
; ARMPL-SAME:         ptr @armpl_svlog10_f32_x
; COMMON-SAME:      ], section "llvm.metadata"

define double @sin_f64(double %in) {
; COMMON-LABEL: @sin_f64(
; SVML:         call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; MASSV:        call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; ACCELERATE:   call double @sin(double %{{.*}})
; LIBMVEC-X86:  call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; SLEEFGNUABI:  call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; ARMPL:        call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; No mapping of "sin" to a vector function for Accelerate.
; ACCELERATE-NOT:  _ZGV_LLVM_{{.*}}_sin({{.*}})
  %call = tail call double @sin(double %in)
  ret double %call
}

declare double @sin(double) #0

define float @call_llvm.log10.f32(float %in) {
; COMMON-LABEL: @call_llvm.log10.f32(
; SVML:         call float @llvm.log10.f32(float %{{.*}})
; LIBMVEC-X86:  call float @llvm.log10.f32(float %{{.*}})
; MASSV:        call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; ACCELERATE:   call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; SLEEFGNUABI:  call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; ARMPL:        call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; No mapping of "llvm.log10.f32" to a vector function for SVML.
; SVML-NOT:        _ZGV_LLVM_{{.*}}_llvm.log10.f32({{.*}})
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
; MASSV-SAME:   "_ZGV_LLVM_N2v_sin(__sind2)" }
; MASSV:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; MASSV-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(__log10f4)" }

; ACCELERATE:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; ACCELERATE-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(vlog10f)" }

; LIBMVEC-X86:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; LIBMVEC-X86-SAME:   "_ZGV_LLVM_N2v_sin(_ZGVbN2v_sin),
; LIBMVEC-X86-SAME:   _ZGV_LLVM_N4v_sin(_ZGVdN4v_sin)" }

; SLEEFGNUABI:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N2v_sin(_ZGVnN2v_sin),
; SLEEFGNUABI-SAME:   _ZGVsMxv_sin(_ZGVsMxv_sin)" }
; SLEEFGNUABI:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(_ZGVnN4v_log10f),
; SLEEFGNUABI-SAME:   _ZGVsMxv_llvm.log10.f32(_ZGVsMxv_log10f)" }

; ARMPL:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N2v_sin(armpl_vsinq_f64),
; ARMPL-SAME     _ZGVsMxv_sin(armpl_svsin_f64_x)" }
; ARMPL:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N4v_llvm.log10.f32(armpl_vlog10q_f32),
; ARMPL-SAME     _ZGVsMxv_llvm.log10.f32(armpl_svlog10_f32_x)" }
