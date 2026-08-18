; RUN: opt -passes='inject-openmp-vfabi-mappings' -S < %s | FileCheck %s

;; Verify @llvm.compiler.used contains the vector declarations.
; CHECK-DAG: @llvm.compiler.used = appending global [{{[0-9]+}} x ptr] [{{.*}}ptr @_ZGVnN2v_neon_single{{.*}}], section "llvm.metadata"

;; AArch64 Advanced SIMD (ISA "n"): single variant on a declaration.
declare void @neon_single(double) #0
; CHECK-DAG: declare aarch64_vector_pcs void @_ZGVnN2v_neon_single(<2 x double>)
; CHECK-DAG: attributes #0 = {{{.*}}"vector-function-abi-variant"="_ZGVnN2v_neon_single(_ZGVnN2v_neon_single)"{{.*}}}

;; AArch64 Advanced SIMD: multiple variants on the same function.
declare float @neon_multi(float, float) #1
; CHECK-DAG: declare aarch64_vector_pcs <4 x float> @_ZGVnN4vv_neon_multi(<4 x float>, <4 x float>)
; CHECK-DAG: declare aarch64_vector_pcs <2 x float> @_ZGVnN2vv_neon_multi(<2 x float>, <2 x float>)
; CHECK-DAG: attributes #1 = {{{.*}}"vector-function-abi-variant"="{{.*}}_ZGVnN2vv_neon_multi(_ZGVnN2vv_neon_multi){{.*}}_ZGVnN4vv_neon_multi(_ZGVnN4vv_neon_multi){{.*}}"{{.*}}}

;; AArch64 Advanced SIMD: function definition (not just declaration).
define void @neon_def(i32 %x) #2 {
entry:
  ret void
}
; CHECK-DAG: declare aarch64_vector_pcs void @_ZGVnN4v_neon_def(<4 x i32>)
; CHECK-DAG: attributes #2 = {{{.*}}"vector-function-abi-variant"="_ZGVnN4v_neon_def(_ZGVnN4v_neon_def)"{{.*}}}

;; AArch64 Advanced SIMD: linear parameter with stride.
declare void @neon_linear(ptr, i32) #3
; CHECK-DAG: declare aarch64_vector_pcs void @_ZGVnN4l2v_neon_linear(ptr, <4 x i32>)
; CHECK-DAG: attributes #3 = {{{.*}}"vector-function-abi-variant"="_ZGVnN4l2v_neon_linear(_ZGVnN4l2v_neon_linear)"{{.*}}}

;; AArch64 Advanced SIMD: masked variant (ISA "n", Mask "M").
declare double @neon_masked(double) #4
; CHECK-DAG: declare aarch64_vector_pcs <2 x double> @_ZGVnM2v_neon_masked(<2 x double>, <2 x i1>)
; CHECK-DAG: attributes #4 = {{{.*}}"vector-function-abi-variant"="_ZGVnM2v_neon_masked(_ZGVnM2v_neon_masked)"{{.*}}}

;; AArch64 SVE (ISA "s"): uses AArch64_SVE_VectorCall calling convention.
declare double @sve_fn(double) #5
; CHECK-DAG: declare aarch64_sve_vector_pcs <vscale x 2 x double> @_ZGVsMxv_sve_fn(<vscale x 2 x double>, <vscale x 2 x i1>)
; CHECK-DAG: attributes #5 = {{{.*}}"vector-function-abi-variant"="_ZGVsMxv_sve_fn(_ZGVsMxv_sve_fn)"{{.*}}}

;; x86 SSE (ISA "b"): uses default calling convention.
declare double @sse_fn(double) #6
; CHECK-DAG: declare <2 x double> @_ZGVbN2v_sse_fn(<2 x double>)
; CHECK-DAG: attributes #6 = {{{.*}}"vector-function-abi-variant"="_ZGVbN2v_sse_fn(_ZGVbN2v_sse_fn)"{{.*}}}

;; x86 AVX2 (ISA "d"): uses default calling convention.
declare double @avx2_fn(double) #7
; CHECK-DAG: declare <4 x double> @_ZGVdN4v_avx2_fn(<4 x double>)
; CHECK-DAG: attributes #7 = {{{.*}}"vector-function-abi-variant"="_ZGVdN4v_avx2_fn(_ZGVdN4v_avx2_fn)"{{.*}}}

;; x86 AVX512 (ISA "e"): uses default calling convention.
declare double @avx512_fn(double) #8
; CHECK-DAG: declare <8 x double> @_ZGVeN8v_avx512_fn(<8 x double>)
; CHECK-DAG: attributes #8 = {{{.*}}"vector-function-abi-variant"="_ZGVeN8v_avx512_fn(_ZGVeN8v_avx512_fn)"{{.*}}}

;; No _ZGV attrs: function is not modified.
declare void @no_zgv(double) #9
; CHECK-DAG: attributes #9 = { noinline }

;; Pre-existing vector-function-abi-variant with additional _ZGV attr:
;; existing mappings are preserved and the new mapping is appended.
declare void @already_has(double) #10
; CHECK-DAG: declare aarch64_vector_pcs void @_ZGVnN2v_already_has_extra(<2 x double>)
; CHECK-DAG: attributes #10 = {{{.*}}"vector-function-abi-variant"="_ZGVnN2v_already_has(_ZGVnN2v_already_has),_ZGVnN2v_already_has_extra(_ZGVnN2v_already_has_extra)"{{.*}}}

;; Malformed _ZGV attr: demangling fails, attr is kept; valid _ZGV is processed.
declare void @malformed(double) #11
; CHECK-DAG: declare aarch64_vector_pcs void @_ZGVnN2v_malformed(<2 x double>)
; CHECK-DAG: attributes #11 = {{{.*}}"_ZGV_malformed_name"{{.*}}"vector-function-abi-variant"="_ZGVnN2v_malformed(_ZGVnN2v_malformed)"{{.*}}}

;; Scalar function with nounwind — vector decl should inherit it.
declare void @with_attrs(double) #12
; CHECK-DAG: declare aarch64_vector_pcs void @_ZGVnN2v_with_attrs(<2 x double>) [[ATTRS:#[0-9]+]]
; CHECK-DAG: attributes [[ATTRS]] = { nounwind }

attributes #0 = { "_ZGVnN2v_neon_single" }
attributes #1 = { "_ZGVnN4vv_neon_multi" "_ZGVnN2vv_neon_multi" }
attributes #2 = { "_ZGVnN4v_neon_def" }
attributes #3 = { "_ZGVnN4l2v_neon_linear" }
attributes #4 = { "_ZGVnM2v_neon_masked" }
attributes #5 = { "_ZGVsMxv_sve_fn" }
attributes #6 = { "_ZGVbN2v_sse_fn" }
attributes #7 = { "_ZGVdN4v_avx2_fn" }
attributes #8 = { "_ZGVeN8v_avx512_fn" }
attributes #9 = { noinline }
attributes #10 = { "vector-function-abi-variant"="_ZGVnN2v_already_has(_ZGVnN2v_already_has)" "_ZGVnN2v_already_has_extra" }
attributes #11 = { "_ZGV_malformed_name" "_ZGVnN2v_malformed" }
attributes #12 = { nounwind "_ZGVnN2v_with_attrs" }
