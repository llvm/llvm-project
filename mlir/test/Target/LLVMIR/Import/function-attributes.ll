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
define void @func_arg_attrs(
    ptr byval(i64) %arg0,
    ptr byref(i64) %arg1,
    ptr sret(i64) %arg2,
    ptr inalloca(i64) %arg3) {
  ret void
}

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
