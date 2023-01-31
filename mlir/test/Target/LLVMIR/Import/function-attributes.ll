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

; CHECK-LABEL: @func_readnone
; CHECK-SAME:  attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>}
; CHECK:   llvm.return
define void @func_readnone() readnone {
  ret void
}

; CHECK-LABEL: @func_readnone_indirect
; CHECK-SAME:  attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>}
declare void @func_readnone_indirect() #0
attributes #0 = { readnone }

; // -----

; CHECK-LABEL: @func_arg_attrs
; CHECK-SAME:  !llvm.ptr {llvm.byval = i64}
; CHECK-SAME:  !llvm.ptr {llvm.byref = i64}
; CHECK-SAME:  !llvm.ptr {llvm.sret = i64}
; CHECK-SAME:  !llvm.ptr {llvm.inalloca = i64}
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
define ptr @func_arg_attrs(
    ptr byval(i64) %arg0,
    ptr byref(i64) %arg1,
    ptr sret(i64) %arg2,
    ptr inalloca(i64) %arg3,
    ptr noalias %arg4,
    ptr readonly %arg5,
    ptr nest %arg6,
    i32 signext %arg7,
    i64 zeroext %arg8,
    ptr align(64) noundef %arg9,
    ptr dereferenceable(12) %arg10,
    ptr dereferenceable_or_null(42) %arg11,
    double inreg %arg12,
    ptr nocapture %arg13,
    ptr nofree %arg14,
    ptr nonnull %arg15,
    ptr preallocated(double) %arg16,
    ptr returned %arg17,
    ptr alignstack(32) %arg18,
    ptr writeonly %arg19) {
  ret ptr %arg17
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

; CHECK-LABEL: @entry_count
; CHECK-SAME:  attributes {function_entry_count = 4242 : i64}
define void @entry_count() !prof !1 {
  ret void
}

!1 = !{!"function_entry_count", i64 4242}

; // -----

; CHECK-LABEL: @func_memory
; CHECK-SAME:  attributes {memory = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = readwrite>}
; CHECK:   llvm.return
define void @func_memory() memory(readwrite, argmem: none) {
  ret void
}

; // -----

; CHECK-LABEL: @passthrough_combined
; CHECK-SAME: attributes {passthrough = [
; CHECK-DAG: ["alignstack", "16"]
; CHECK-DAG: "noinline"
; CHECK-DAG: "probe-stack"
; CHECK-DAG: ["alloc-family", "malloc"]
; CHECK:   llvm.return
define void @passthrough_combined() alignstack(16) noinline "probe-stack" "alloc-family"="malloc" {
  ret void
}

// -----

; CHECK-LABEL: @passthrough_string_only
; CHECK-SAME: attributes {passthrough = ["no-enum-attr"]}
; CHECK:   llvm.return
define void @passthrough_string_only() "no-enum-attr" {
  ret void
}
