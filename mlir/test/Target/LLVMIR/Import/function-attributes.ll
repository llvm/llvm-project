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
; CHECK-SAME:  attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>}
; CHECK:   llvm.return
define void @func_readnone() readnone {
  ret void
}

; CHECK-LABEL: @func_readnone_indirect
; CHECK-SAME:  attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>}
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
; CHECK-SAME:  attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = readwrite, errnoMem = readwrite, targetMem0 = readwrite, targetMem1 = readwrite>}
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

; CHECK-LABEL: @func_attr_denormal_fp_math_ieee(){{$}}
declare void @func_attr_denormal_fp_math_ieee() denormal_fpenv(ieee)

; // -----

; CHECK-LABEL: @func_attr_denormal_fp_math_f32_preserve_sign
; CHECK-SAME: attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee, float_output_mode = preservesign, float_input_mode = preservesign>}
declare void @func_attr_denormal_fp_math_f32_preserve_sign() denormal_fpenv(float: preservesign)

; // -----

; CHECK-LABEL: @func_attr_mixed_denormal_modes
; CHECK: attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = dynamic, default_input_mode = preservesign, float_output_mode = preservesign, float_input_mode = dynamic>}
declare void @func_attr_mixed_denormal_modes() denormal_fpenv(dynamic|preservesign, float: preservesign|dynamic)

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

; CHECK-LABEL: @inlinehint_attribute
; CHECK-SAME: attributes {inline_hint}
declare void @inlinehint_attribute() inlinehint

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

// -----

; CHECK-LABEL: @noreturn_attribute
; CHECK-SAME: attributes {noreturn}
declare void @noreturn_attribute() noreturn

// -----

; CHECK-LABEL: @returnstwice_attribute
; CHECK-SAME: attributes {returns_twice}
declare void @returnstwice_attribute() returns_twice

// -----

; CHECK-LABEL: @hot_attribute
; CHECK-SAME: attributes {hot}
declare void @hot_attribute() hot

// -----

; CHECK-LABEL: @cold_attribute
; CHECK-SAME: attributes {cold}
declare void @cold_attribute() cold

// -----

; CHECK-LABEL: @noduplicate_attribute
; CHECK-SAME: attributes {noduplicate}
declare void @noduplicate_attribute() noduplicate

// -----

; CHECK-LABEL: @no_caller_saved_registers_attribute
; CHECK-SAME: attributes {no_caller_saved_registers}
declare void @no_caller_saved_registers_attribute () "no_caller_saved_registers"

// -----

; CHECK-LABEL: @nocallback_attribute
; CHECK-SAME: attributes {nocallback}
declare void @nocallback_attribute() nocallback

// -----

; CHECK-LABEL: @modular_format_attribute
; CHECK-SAME: attributes {modular_format = "Ident,1,1,Foo,Bar"}
declare void @modular_format_attribute(i32) "modular-format" = "Ident,1,1,Foo,Bar"

// -----

; CHECK-LABEL: @no_builtins_all
; CHECK-SAME: attributes {nobuiltins = []}
declare void @no_builtins_all() "no-builtins"

// -----

; CHECK-LABEL: @no_builtins_2
; CHECK-SAME: attributes {nobuiltins = ["asdf", "defg"]}
declare void @no_builtins_2() "no-builtin-asdf" "no-builtin-defg"

// -----

; CHECK-LABEL: @alloc_size_1
; CHECK-SAME: attributes {allocsize = array<i32: 0>}
declare void @alloc_size_1(i32) allocsize(0)

// -----

; CHECK-LABEL: @alloc_size_2
; CHECK-SAME: attributes {allocsize = array<i32: 0, 1>}
declare void @alloc_size_2(i32, i32) allocsize(0, 1)

// -----

; CHECK-LABEL: @minsize
; CHECK-SAME: attributes {minsize}
declare void @minsize() minsize

// -----

; CHECK-LABEL: @optsize
; CHECK-SAME: attributes {optsize}
declare void @optsize() optsize

// -----

; CHECK-LABEL: @save_reg_params
; CHECK-SAME: attributes {save_reg_params}
declare void @save_reg_params() "save-reg-params"

// -----

; CHECK-LABEL: @zero_call_used_regs
; CHECK-SAME: attributes {zero_call_used_regs = "skip"}
declare void @zero_call_used_regs() "zero-call-used-regs"="skip"

// -----

; Note: the 'default-func-attrs' aren't recoverable due to the way they lower
; to LLVM-IR, so they are handled on import as passthrough attributes.
; CHECK-LABEL: @default_func_attrs
; CHECK-SAME: attributes {passthrough = {{\[}}["key", "value"], "keyOnly"]}
declare void @default_func_attrs() "key"="value" "keyOnly"

// -----

; expected-warning @unknown {{'preallocated' attribute is invalid on current operation, skipping it}}
declare void @test() preallocated(i32)
