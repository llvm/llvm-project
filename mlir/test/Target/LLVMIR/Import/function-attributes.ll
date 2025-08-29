; RUN: mlir-translate -import-llvm -split-input-file %s --verify-diagnostics | FileCheck %s

; CHECK: llvm.func internal @func_internal
define internal void @func_internal() {
  ret void
}

; CHECK: llvm.func internal spir_funccc @spir_func_internal()
define internal spir_func void @spir_func_internal() {
  ret void
}

; // -----

; Ensure that we have dso_local.
; CHECK: llvm.func @dsolocal_func()
; CHECK-SAME: attributes {dso_local}
define dso_local void @dsolocal_func() {
  ret void
}

; // -----

; CHECK-LABEL: @func_readnone
; CHECK-SAME:  attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>}
; CHECK:   llvm.return
define void @func_readnone() readnone {
  ret void
}

; CHECK-LABEL: @func_readnone_indirect
; CHECK-SAME:  attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>}
declare void @func_readnone_indirect() #0
attributes #0 = { readnone }

; // -----

; CHECK-LABEL: @func_arg_attrs
; CHECK-SAME:  !llvm.ptr {llvm.byval = i64}
; CHECK-SAME:  !llvm.ptr {llvm.byref = i64}
; CHECK-SAME:  !llvm.ptr {llvm.noalias}
; CHECK-SAME:  !llvm.ptr {llvm.readonly}
; CHECK-SAME:  !llvm.ptr {llvm.nest}
; CHECK-SAME:  i32 {llvm.signext}
; CHECK-SAME:  i64 {llvm.zeroext}
; CHECK-SAME:  !llvm.ptr {llvm.align = 64 : i64, llvm.noundef}
; CHECK-SAME:  !llvm.ptr {llvm.dereferenceable = 12 : i64}
; CHECK-SAME:  !llvm.ptr {llvm.dereferenceable_or_null = 42 : i64}
; CHECK-SAME:  f64 {llvm.inreg}
; CHECK-SAME:  !llvm.ptr {llvm.nocapture}
; CHECK-SAME:  !llvm.ptr {llvm.nofree}
; CHECK-SAME:  !llvm.ptr {llvm.nonnull}
; CHECK-SAME:  !llvm.ptr {llvm.preallocated = f64}
; CHECK-SAME:  !llvm.ptr {llvm.returned}
; CHECK-SAME:  !llvm.ptr {llvm.alignstack = 32 : i64}
; CHECK-SAME:  !llvm.ptr {llvm.writeonly}
; CHECK-SAME:  i64 {llvm.range = #llvm.constant_range<i64, 0, 4097>}
define ptr @func_arg_attrs(
    ptr byval(i64) %arg0,
    ptr byref(i64) %arg1,
    ptr noalias %arg4,
    ptr readonly %arg5,
    ptr nest %arg6,
    i32 signext %arg7,
    i64 zeroext %arg8,
    ptr align(64) noundef %arg9,
    ptr dereferenceable(12) %arg10,
    ptr dereferenceable_or_null(42) %arg11,
    double inreg %arg12,
    ptr captures(none) %arg13,
    ptr nofree %arg14,
    ptr nonnull %arg15,
    ptr preallocated(double) %arg16,
    ptr returned %arg17,
    ptr alignstack(32) %arg18,
    ptr writeonly %arg19,
    i64 range(i64 0, 4097) %arg20) {
  ret ptr %arg17
}

; CHECK-LABEL: @sret
; CHECK-SAME:  !llvm.ptr {llvm.sret = i64}
define void @sret(ptr sret(i64) %arg0) {
  ret void
}

; CHECK-LABEL: @inalloca
; CHECK-SAME:  !llvm.ptr {llvm.inalloca = i64}
define void @inalloca(ptr inalloca(i64) %arg0) {
  ret void
}

; CHECK-LABEL: @allocator
; CHECK-SAME:  i64 {llvm.allocalign}
; CHECK-SAME:  ptr {llvm.allocptr}
declare ptr @allocator(i64 allocalign, ptr allocptr)

; // -----

; CHECK-LABEL: @func_res_attr_noalias
; CHECK-SAME:  !llvm.ptr {llvm.noalias}
declare noalias ptr @func_res_attr_noalias()

; // -----

; CHECK-LABEL: @func_res_attr_nonnull
; CHECK-SAME:  !llvm.ptr {llvm.nonnull}
declare nonnull ptr @func_res_attr_nonnull()

; // -----

; CHECK-LABEL: @func_res_attr_signext
; CHECK-DAG: llvm.noundef
; CHECK-DAG: llvm.signext
declare noundef signext i32 @func_res_attr_signext()

; // -----

; CHECK-LABEL: @func_res_attr_zeroext
; CHECK-SAME:  i32 {llvm.zeroext}
declare zeroext i32 @func_res_attr_zeroext()

; // -----

; CHECK-LABEL: @func_res_attr_align
; CHECK-SAME:  !llvm.ptr {llvm.align = 16 : i64}
declare align(16) ptr @func_res_attr_align()

; // -----

; CHECK-LABEL: @func_res_attr_noundef
; CHECK-SAME:  !llvm.ptr {llvm.noundef}
declare noundef ptr @func_res_attr_noundef()

; // -----

; CHECK-LABEL: @func_res_attr_dereferenceable
; CHECK-SAME:  !llvm.ptr {llvm.dereferenceable = 42 : i64}
declare dereferenceable(42) ptr @func_res_attr_dereferenceable()

; // -----

; CHECK-LABEL: @func_res_attr_dereferenceable_or_null
; CHECK-SAME:  !llvm.ptr {llvm.dereferenceable_or_null = 42 : i64}
declare dereferenceable_or_null(42) ptr @func_res_attr_dereferenceable_or_null()

; // -----

; CHECK-LABEL: @func_res_attr_inreg
; CHECK-SAME:  !llvm.ptr {llvm.inreg}
declare inreg ptr @func_res_attr_inreg()

; // -----

; CHECK-LABEL: @func_res_attr_range
; CHECK-SAME:  (i64 {llvm.range = #llvm.constant_range<i64, 0, 4097>})
declare range(i64 0, 4097) i64 @func_res_attr_range()

; // -----

; CHECK-LABEL: @entry_count
; CHECK-SAME:  attributes {function_entry_count = 4242 : i64}
define void @entry_count() !prof !1 {
  ret void
}

!1 = !{!"function_entry_count", i64 4242}

; // -----

; CHECK-LABEL: @func_memory
; CHECK-SAME:  attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = readwrite>}
; CHECK:   llvm.return
define void @func_memory() memory(readwrite, argmem: none) {
  ret void
}

; // -----

; CHECK-LABEL: @passthrough_combined
; CHECK-SAME: attributes {passthrough = [
; CHECK-DAG: ["alignstack", "16"]
; CHECK-DAG: "probe-stack"
; CHECK-DAG: ["alloc-family", "malloc"]
; CHECK:   llvm.return
define void @passthrough_combined() alignstack(16) "probe-stack" "alloc-family"="malloc" {
  ret void
}

// -----

; CHECK-LABEL: @passthrough_string_only
; CHECK-SAME: attributes {passthrough = ["no-enum-attr"]}
; CHECK:   llvm.return
define void @passthrough_string_only() "no-enum-attr" {
  ret void
}

// -----

; CHECK-LABEL: llvm.func hidden @hidden()
define hidden void @hidden() {
  ret void
}

// -----

; CHECK-LABEL: llvm.func protected @protected()
define protected void @protected() {
  ret void
}

// -----

; CHECK-LABEL: @streaming_func
; CHECK-SAME: attributes {arm_streaming}
define void @streaming_func() "aarch64_pstate_sm_enabled" {
  ret void
}

// -----

; CHECK-LABEL: @locally_streaming_func
; CHECK-SAME: attributes {arm_locally_streaming}
define void @locally_streaming_func() "aarch64_pstate_sm_body" {
  ret void
}

// -----

; CHECK-LABEL: @streaming_compatible_func
; CHECK-SAME: attributes {arm_streaming_compatible}
define void @streaming_compatible_func() "aarch64_pstate_sm_compatible" {
  ret void
}

// -----

; CHECK-LABEL: @arm_new_za_func
; CHECK-SAME: attributes {arm_new_za}
define void @arm_new_za_func() "aarch64_new_za" {
  ret void
}


; CHECK-LABEL: @arm_in_za_func
; CHECK-SAME: attributes {arm_in_za}
define void @arm_in_za_func() "aarch64_in_za" {
  ret void
}

; CHECK-LABEL: @arm_out_za_func
; CHECK-SAME: attributes {arm_out_za}
define void @arm_out_za_func() "aarch64_out_za" {
  ret void
}

; CHECK-LABEL: @arm_inout_za_func
; CHECK-SAME: attributes {arm_inout_za}
define void @arm_inout_za_func() "aarch64_inout_za" {
  ret void
}

; CHECK-LABEL: @arm_preserves_za_func
; CHECK-SAME: attributes {arm_preserves_za}
define void @arm_preserves_za_func() "aarch64_preserves_za" {
  ret void
}

// -----

; CHECK-LABEL: @section_func
; CHECK-SAME: attributes {section = ".section.name"}
define void @section_func() section ".section.name" {
  ret void
}

// -----

; CHECK-LABEL: local_unnamed_addr @local_unnamed_addr_func
define void @local_unnamed_addr_func() local_unnamed_addr {
  ret void
}

// -----

; CHECK-LABEL: unnamed_addr @unnamed_addr_func
declare void @unnamed_addr_func() unnamed_addr

// -----

; CHECK-LABEL: @align_func
; CHECK-SAME: attributes {alignment = 2 : i64}
define void @align_func() align 2 {
  ret void
}

// -----

; CHECK-LABEL: @align_decl
; CHECK-SAME: attributes {alignment = 64 : i64}
declare void @align_decl() align 64

; // -----

; CHECK-LABEL: @func_attr_unsafe_fp_math_true
; CHECK-SAME: attributes {unsafe_fp_math = true}
declare void @func_attr_unsafe_fp_math_true() "unsafe-fp-math"="true"

; // -----

; CHECK-LABEL: @func_attr_unsafe_fp_math_false
; CHECK-SAME: attributes {unsafe_fp_math = false}
declare void @func_attr_unsafe_fp_math_false() "unsafe-fp-math"="false"

; // -----

; CHECK-LABEL: @func_attr_no_infs_fp_math_true
; CHECK-SAME: attributes {no_infs_fp_math = true}
declare void @func_attr_no_infs_fp_math_true() "no-infs-fp-math"="true"

; // -----

; CHECK-LABEL: @func_attr_no_infs_fp_math_false
; CHECK-SAME: attributes {no_infs_fp_math = false}
declare void @func_attr_no_infs_fp_math_false() "no-infs-fp-math"="false"

; // -----

; CHECK-LABEL: @func_attr_no_nans_fp_math_true
; CHECK-SAME: attributes {no_nans_fp_math = true}
declare void @func_attr_no_nans_fp_math_true() "no-nans-fp-math"="true"

; // -----

; CHECK-LABEL: @func_attr_no_nans_fp_math_false
; CHECK-SAME: attributes {no_nans_fp_math = false}
declare void @func_attr_no_nans_fp_math_false() "no-nans-fp-math"="false"

; // -----

; CHECK-LABEL: @func_attr_no_signed_zeros_fp_math_true
; CHECK-SAME: attributes {no_signed_zeros_fp_math = true}
declare void @func_attr_no_signed_zeros_fp_math_true() "no-signed-zeros-fp-math"="true"

; // -----

; CHECK-LABEL: @func_attr_no_signed_zeros_fp_math_false
; CHECK-SAME: attributes {no_signed_zeros_fp_math = false}
declare void @func_attr_no_signed_zeros_fp_math_false() "no-signed-zeros-fp-math"="false"

; // -----

; CHECK-LABEL: @func_attr_denormal_fp_math_ieee
; CHECK-SAME: attributes {denormal_fp_math = "ieee"}
declare void @func_attr_denormal_fp_math_ieee() "denormal-fp-math"="ieee"

; // -----

; CHECK-LABEL: @func_attr_denormal_fp_math_f32_preserve_sign
; CHECK-SAME: attributes {denormal_fp_math_f32 = "preserve-sign"}
declare void @func_attr_denormal_fp_math_f32_preserve_sign() "denormal-fp-math-f32"="preserve-sign"

; // -----

; CHECK-LABEL: @func_attr_fp_contract_fast
; CHECK-SAME: attributes {fp_contract = "fast"}
declare void @func_attr_fp_contract_fast() "fp-contract"="fast"

// -----

; CHECK-LABEL: @func_attr_instrument_function_entry
; CHECK-SAME: attributes {instrument_function_entry = "__cyg_profile_func_enter"}
declare void @func_attr_instrument_function_entry() "instrument-function-entry"="__cyg_profile_func_enter"

// -----

; CHECK-LABEL: @func_attr_instrument_function_exit
; CHECK-SAME: attributes {instrument_function_exit = "__cyg_profile_func_exit"}
declare void @func_attr_instrument_function_exit() "instrument-function-exit"="__cyg_profile_func_exit"

// -----

; CHECK-LABEL: @noinline_attribute
; CHECK-SAME: attributes {no_inline}
declare void @noinline_attribute() noinline

// -----

; CHECK-LABEL: @alwaysinline_attribute
; CHECK-SAME: attributes {always_inline}
declare void @alwaysinline_attribute() alwaysinline

// -----

; CHECK-LABEL: @optnone_attribute
; CHECK-SAME: attributes {no_inline, optimize_none}
declare void @optnone_attribute() noinline optnone

// -----

; CHECK-LABEL: @convergent_attribute
; CHECK-SAME: attributes {convergent}
declare void @convergent_attribute() convergent

// -----

; CHECK-LABEL: @nounwind_attribute
; CHECK-SAME: attributes {no_unwind}
declare void @nounwind_attribute() nounwind

// -----

; CHECK-LABEL: @willreturn_attribute
; CHECK-SAME: attributes {will_return}
declare void @willreturn_attribute() willreturn
