; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=SVML -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,SVML
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=AMDLIBM -passes=inject-tli-mappings -S < %s | FileCheck %s  --check-prefixes=COMMON,AMDLIBM
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
; AMDLIBM-SAME:     [6 x ptr] [
; AMDLIBM-SAME:       ptr @amd_vrd2_sin,
; AMDLIBM-SAME:       ptr @amd_vrd4_sin,
; AMDLIBM-SAME:       ptr @amd_vrd8_sin,
; AMDLIBM-SAME:       ptr @amd_vrs4_log10f,
; AMDLIBM-SAME:       ptr @amd_vrs8_log10f,
; AMDLIBM-SAME:       ptr @amd_vrs16_log10f
; MASSV-SAME:       [2 x ptr] [
; MASSV-SAME:         ptr @__sind2,
; MASSV-SAME:         ptr @__log10f4
; ACCELERATE-SAME:  [1 x ptr] [
; ACCELERATE-SAME:    ptr @vlog10f
; LIBMVEC-X86-SAME: [2 x ptr] [
; LIBMVEC-X86-SAME:   ptr @_ZGVbN2v_sin,
; LIBMVEC-X86-SAME:   ptr @_ZGVdN4v_sin
; SLEEFGNUABI-SAME: [16 x ptr] [
; SLEEFGNUABI-SAME:   ptr @_ZGVnN2vl8_modf,
; SLEEFGNUABI-SAME:   ptr @_ZGVsNxvl8_modf,
; SLEEFGNUABI-SAME:   ptr @_ZGVnN4vl4_modff,
; SLEEFGNUABI-SAME:   ptr @_ZGVsNxvl4_modff,
; SLEEFGNUABI-SAME:   ptr @_ZGVnN2v_sin,
; SLEEFGNUABI-SAME:   ptr @_ZGVsMxv_sin,
; SLEEFGNUABI-SAME:   ptr @_ZGVnN2vl8l8_sincos,
; SLEEFGNUABI-SAME:   ptr @_ZGVsNxvl8l8_sincos,
; SLEEFGNUABI-SAME:   ptr @_ZGVnN4vl4l4_sincosf,
; SLEEFGNUABI-SAME:   ptr @_ZGVsNxvl4l4_sincosf,
; SLEEFGNUABI-SAME:   ptr @_ZGVnN2vl8l8_sincospi,
; SLEEFGNUABI-SAME:   ptr @_ZGVsNxvl8l8_sincospi,
; SLEEFGNUABI-SAME:   ptr @_ZGVnN4vl4l4_sincospif,
; SLEEFGNUABI-SAME:   ptr @_ZGVsNxvl4l4_sincospif,
; SLEEFGNUABI_SAME;   ptr @_ZGVnN4v_log10f,
; SLEEFGNUABI-SAME:   ptr @_ZGVsMxv_log10f
; ARMPL-SAME:       [10 x ptr] [
; ARMPL-SAME:         ptr @armpl_vmodfq_f64,
; ARMPL-SAME:         ptr @armpl_vmodfq_f32,
; ARMPL-SAME:         ptr @armpl_vsinq_f64,
; ARMPL-SAME:         ptr @armpl_svsin_f64_x,
; ARMPL-SAME:         ptr @armpl_vsincosq_f64,
; ARMPL-SAME:         ptr @armpl_vsincosq_f32,
; ARMPL-SAME:         ptr @armpl_vsincospiq_f64,
; ARMPL-SAME:         ptr @armpl_vsincospiq_f32,
; ARMPL-SAME:         ptr @armpl_vlog10q_f32,
; ARMPL-SAME:         ptr @armpl_svlog10_f32_x
; COMMON-SAME:      ], section "llvm.metadata"

define double @modf_f64(double %in, ptr %iptr) {
; COMMON-LABEL: @modf_f64(
; SLEEFGNUABI:  call double @modf(double %{{.*}}, ptr %{{.*}}) #[[MODF:[0-9]+]]
; ARMPL:        call double @modf(double %{{.*}}, ptr %{{.*}}) #[[MODF:[0-9]+]]
  %call = tail call double @modf(double %in, ptr %iptr)
  ret double %call
}

declare double @modf(double, ptr) #0

define float @modf_f32(float %in, ptr %iptr) {
; COMMON-LABEL: @modf_f32(
; SLEEFGNUABI:  call float @modff(float %{{.*}}, ptr %{{.*}}) #[[MODFF:[0-9]+]]
; ARMPL:        call float @modff(float %{{.*}}, ptr %{{.*}}) #[[MODFF:[0-9]+]]
  %call = tail call float @modff(float %in, ptr %iptr)
  ret float %call
}

declare float @modff(float, ptr) #0

define double @sin_f64(double %in) {
; COMMON-LABEL: @sin_f64(
; SVML:         call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
; AMDLIBM:      call double @sin(double %{{.*}}) #[[SIN:[0-9]+]]
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

define void @sincos_f64(double %in, ptr %sin, ptr %cos) {
; COMMON-LABEL: @sincos_f64(
; SLEEFGNUABI:  call void @sincos(double %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOS:[0-9]+]]
; ARMPL:        call void @sincos(double %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOS:[0-9]+]]
  call void @sincos(double %in, ptr %sin, ptr %cos)
  ret void
}

declare void @sincos(double, ptr, ptr) #0

define void @sincos_f32(float %in, ptr %sin, ptr %cos) {
; COMMON-LABEL: @sincos_f32(
; SLEEFGNUABI:  call void @sincosf(float %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOSF:[0-9]+]]
; ARMPL:        call void @sincosf(float %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOSF:[0-9]+]]
  call void @sincosf(float %in, ptr %sin, ptr %cos)
  ret void
}

declare void @sincosf(float, ptr, ptr) #0

define void @sincospi_f64(double %in, ptr %sin, ptr %cos) {
; COMMON-LABEL: @sincospi_f64(
; SLEEFGNUABI:  call void @sincospi(double %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOSPI:[0-9]+]]
; ARMPL:        call void @sincospi(double %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOSPI:[0-9]+]]
  call void @sincospi(double %in, ptr %sin, ptr %cos)
  ret void
}

declare void @sincospi(double, ptr, ptr) #0

define void @sincospi_f32(float %in, ptr %sin, ptr %cos) {
; COMMON-LABEL: @sincospi_f32(
; SLEEFGNUABI:  call void @sincospif(float %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOSPIF:[0-9]+]]
; ARMPL:        call void @sincospif(float %{{.*}}, ptr %{{.*}}, ptr %{{.*}}) #[[SINCOSPIF:[0-9]+]]
  call void @sincospif(float %in, ptr %sin, ptr %cos)
  ret void
}

declare void @sincospif(float, ptr, ptr) #0

define float @call_llvm.log10.f32(float %in) {
; COMMON-LABEL: @call_llvm.log10.f32(
; SVML:         call float @llvm.log10.f32(float %{{.*}})
; AMDLIBM:      call float @llvm.log10.f32(float %{{.*}})
; LIBMVEC-X86:  call float @llvm.log10.f32(float %{{.*}})
; MASSV:        call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; ACCELERATE:   call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; SLEEFGNUABI:  call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; ARMPL:        call float @llvm.log10.f32(float %{{.*}}) #[[LOG10:[0-9]+]]
; No mapping of "llvm.log10.f32" to a vector function for SVML.
; SVML-NOT:        _ZGV_LLVM_{{.*}}_llvm.log10.f32({{.*}})
; AMDLIBM-NOT:        _ZGV_LLVM_{{.*}}_llvm.log10.f32({{.*}})
; LIBMVEC-X86-NOT: _ZGV_LLVM_{{.*}}_llvm.log10.f32({{.*}})
  %call = tail call float @llvm.log10.f32(float %in)
  ret float %call
}

declare float @llvm.log10.f32(float) #0

; SVML: declare <2 x double> @__svml_sin2(<2 x double>)
; SVML: declare <4 x double> @__svml_sin4(<4 x double>)
; SVML: declare <8 x double> @__svml_sin8(<8 x double>)
; SVML: declare <4 x float> @__svml_log10f4(<4 x float>)
; SVML: declare <8 x float> @__svml_log10f8(<8 x float>)
; SVML: declare <16 x float> @__svml_log10f16(<16 x float>)

; AMDLIBM: declare <2 x double> @amd_vrd2_sin(<2 x double>)
; AMDLIBM: declare <4 x double> @amd_vrd4_sin(<4 x double>)
; AMDLIBM: declare <8 x double> @amd_vrd8_sin(<8 x double>)
; AMDLIBM: declare <4 x float> @amd_vrs4_log10f(<4 x float>)
; AMDLIBM: declare <8 x float> @amd_vrs8_log10f(<8 x float>)
; AMDLIBM: declare <16 x float> @amd_vrs16_log10f(<16 x float>)

; MASSV: declare <2 x double> @__sind2(<2 x double>)
; MASSV: declare <4 x float> @__log10f4(<4 x float>)

; LIBMVEC-X86: declare <2 x double> @_ZGVbN2v_sin(<2 x double>)
; LIBMVEC-X86: declare <4 x double> @_ZGVdN4v_sin(<4 x double>)

; ACCELERATE: declare <4 x float> @vlog10f(<4 x float>)

; SLEEFGNUABI: declare <2 x double> @_ZGVnN2vl8_modf(<2 x double>, ptr)
; SLEEFGNUABI: declare <vscale x 2 x double> @_ZGVsNxvl8_modf(<vscale x 2 x double>, ptr)
; SLEEFGNUABI: declare <4 x float> @_ZGVnN4vl4_modff(<4 x float>, ptr)
; SLEEFGNUABI: declare <vscale x 4 x float> @_ZGVsNxvl4_modff(<vscale x 4 x float>, ptr)
; SLEEFGNUABI: declare <2 x double> @_ZGVnN2v_sin(<2 x double>)
; SLEEFGNUABI: declare <vscale x 2 x double> @_ZGVsMxv_sin(<vscale x 2 x double>, <vscale x 2 x i1>)
; SLEEFGNUABI: declare void @_ZGVnN2vl8l8_sincos(<2 x double>, ptr, ptr)
; SLEEFGNUABI: declare void @_ZGVsNxvl8l8_sincos(<vscale x 2 x double>, ptr, ptr)
; SLEEFGNUABI: declare void @_ZGVnN4vl4l4_sincosf(<4 x float>, ptr, ptr)
; SLEEFGNUABI: declare void @_ZGVsNxvl4l4_sincosf(<vscale x 4 x float>, ptr, ptr)
; SLEEFGNUABI: declare void @_ZGVnN2vl8l8_sincospi(<2 x double>, ptr, ptr)
; SLEEFGNUABI: declare void @_ZGVsNxvl8l8_sincospi(<vscale x 2 x double>, ptr, ptr)
; SLEEFGNUABI: declare void @_ZGVnN4vl4l4_sincospif(<4 x float>, ptr, ptr)
; SLEEFGNUABI: declare void @_ZGVsNxvl4l4_sincospif(<vscale x 4 x float>, ptr, ptr)
; SLEEFGNUABI: declare <4 x float> @_ZGVnN4v_log10f(<4 x float>)
; SLEEFGNUABI: declare <vscale x 4 x float> @_ZGVsMxv_log10f(<vscale x 4 x float>, <vscale x 4 x i1>)

; ARMPL: declare <2 x double> @armpl_vmodfq_f64(<2 x double>, ptr)
; ARMPL: declare <4 x float> @armpl_vmodfq_f32(<4 x float>, ptr)
; ARMPL: declare <2 x double> @armpl_vsinq_f64(<2 x double>)
; ARMPL: declare <vscale x 2 x double> @armpl_svsin_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>)
; ARMPL: declare void @armpl_vsincosq_f64(<2 x double>, ptr, ptr)
; ARMPL: declare void @armpl_vsincosq_f32(<4 x float>, ptr, ptr)
; ARMPL: declare void @armpl_vsincospiq_f64(<2 x double>, ptr, ptr)
; ARMPL: declare void @armpl_vsincospiq_f32(<4 x float>, ptr, ptr)
; ARMPL: declare <4 x float> @armpl_vlog10q_f32(<4 x float>)
; ARMPL: declare <vscale x 4 x float> @armpl_svlog10_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>)

attributes #0 = { nounwind readnone }

; SVML:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; SVML-SAME:   "_ZGV_LLVM_N2v_sin(__svml_sin2),
; SVML-SAME:   _ZGV_LLVM_N4v_sin(__svml_sin4),
; SVML-SAME:   _ZGV_LLVM_N8v_sin(__svml_sin8)" }

; AMDLIBM:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; AMDLIBM-SAME:   "_ZGV_LLVM_N2v_sin(amd_vrd2_sin),
; AMDLIBM-SAME:   _ZGV_LLVM_N4v_sin(amd_vrd4_sin),
; AMDLIBM-SAME:   _ZGV_LLVM_N8v_sin(amd_vrd8_sin)" }

; MASSV:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; MASSV-SAME:   "_ZGV_LLVM_N2v_sin(__sind2)" }
; MASSV:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; MASSV-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(__log10f4)" }

; ACCELERATE:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; ACCELERATE-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(vlog10f)" }

; LIBMVEC-X86:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; LIBMVEC-X86-SAME:   "_ZGV_LLVM_N2v_sin(_ZGVbN2v_sin),
; LIBMVEC-X86-SAME:   _ZGV_LLVM_N4v_sin(_ZGVdN4v_sin)" }

; SLEEFGNUABI:      attributes #[[MODF]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N2vl8_modf(_ZGVnN2vl8_modf),
; SLEEFGNUABI-SAME:   _ZGVsNxvl8_modf(_ZGVsNxvl8_modf)" }
; SLEEFGNUABI:      attributes #[[MODFF]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N4vl4_modff(_ZGVnN4vl4_modff),
; SLEEFGNUABI-SAME:   _ZGVsNxvl4_modff(_ZGVsNxvl4_modff)" }
; SLEEFGNUABI:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N2v_sin(_ZGVnN2v_sin),
; SLEEFGNUABI-SAME:   _ZGVsMxv_sin(_ZGVsMxv_sin)" }
; SLEEFGNUABI:      attributes #[[SINCOS]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N2vl8l8_sincos(_ZGVnN2vl8l8_sincos),
; SLEEFGNUABI-SAME:   _ZGVsNxvl8l8_sincos(_ZGVsNxvl8l8_sincos)" }
; SLEEFGNUABI:      attributes #[[SINCOSF]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N4vl4l4_sincosf(_ZGVnN4vl4l4_sincosf),
; SLEEFGNUABI-SAME:   _ZGVsNxvl4l4_sincosf(_ZGVsNxvl4l4_sincosf)" }
; SLEEFGNUABI:      attributes #[[SINCOSPI]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N2vl8l8_sincospi(_ZGVnN2vl8l8_sincospi),
; SLEEFGNUABI-SAME:   _ZGVsNxvl8l8_sincospi(_ZGVsNxvl8l8_sincospi)" }
; SLEEFGNUABI:      attributes #[[SINCOSPIF]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N4vl4l4_sincospif(_ZGVnN4vl4l4_sincospif),
; SLEEFGNUABI-SAME:   _ZGVsNxvl4l4_sincospif(_ZGVsNxvl4l4_sincospif)" }
; SLEEFGNUABI:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; SLEEFGNUABI-SAME:   "_ZGV_LLVM_N4v_llvm.log10.f32(_ZGVnN4v_log10f),
; SLEEFGNUABI-SAME:   _ZGVsMxv_llvm.log10.f32(_ZGVsMxv_log10f)" }

; ARMPL:      attributes #[[MODF]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N2vl8_modf(armpl_vmodfq_f64)" }
; ARMPL:      attributes #[[MODFF]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N4vl4_modff(armpl_vmodfq_f32)" }
; ARMPL:      attributes #[[SIN]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N2v_sin(armpl_vsinq_f64),
; ARMPL-SAME     _ZGVsMxv_sin(armpl_svsin_f64_x)" }
; ARMPL:      attributes #[[SINCOS]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N2vl8l8_sincos(armpl_vsincosq_f64)" }
; ARMPL:      attributes #[[SINCOSF]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N4vl4l4_sincosf(armpl_vsincosq_f32)" }
; ARMPL:      attributes #[[SINCOSPI]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N2vl8l8_sincospi(armpl_vsincospiq_f64)" }
; ARMPL:      attributes #[[SINCOSPIF]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N4vl4l4_sincospif(armpl_vsincospiq_f32)" }
; ARMPL:      attributes #[[LOG10]] = { "vector-function-abi-variant"=
; ARMPL-SAME:    "_ZGV_LLVM_N4v_llvm.log10.f32(armpl_vlog10q_f32),
; ARMPL-SAME     _ZGVsMxv_llvm.log10.f32(armpl_svlog10_f32_x)" }
