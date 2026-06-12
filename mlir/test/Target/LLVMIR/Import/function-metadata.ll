; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: llvm.func @repeated_type_metadata
; CHECK-SAME: function_metadata
; CHECK-SAME: #llvm.func_metadata<"type", <#llvm.md_const<0 : i64>, #llvm.md_string<"typeid0">>>
; CHECK-SAME: #llvm.func_metadata<"type", <#llvm.md_const<0 : i64>, #llvm.md_string<"typeid1">>>
define void @repeated_type_metadata() !type !0 !type !1 {
  ret void
}

!0 = !{i64 0, !"typeid0"}
!1 = !{i64 0, !"typeid1"}

; // -----

; CHECK-LABEL: llvm.func @declaration_metadata
; CHECK-SAME: function_metadata
; CHECK-SAME: #llvm.func_metadata<"annotation", <#llvm.md_string<"declaration annotation">>>
declare !annotation !0 void @declaration_metadata()

!0 = !{!"declaration annotation"}

; // -----

declare void @callee()

; CHECK-LABEL: llvm.func @function_ref_metadata
; CHECK-SAME: function_metadata
; CHECK-SAME: #llvm.func_metadata<"callees", <#llvm.md_value<@callee>>>
define void @function_ref_metadata() !callees !0 {
  ret void
}

!0 = !{ptr @callee}

; // -----

define void @alias_target() {
  ret void
}
@alias = alias void (), ptr @alias_target

; CHECK-LABEL: llvm.func @alias_ref_metadata
; CHECK-SAME: function_metadata
; CHECK-SAME: #llvm.func_metadata<"callees", <#llvm.md_value<@alias>>>
define void @alias_ref_metadata() !callees !0 {
  ret void
}

!0 = !{ptr @alias}

; // -----

@global = global i32 0

; CHECK-LABEL: llvm.func @global_ref_metadata
; CHECK-SAME: function_metadata
; CHECK-SAME: #llvm.func_metadata<"callees", <#llvm.md_value<@global>>>
define void @global_ref_metadata() !callees !0 {
  ret void
}

!0 = !{ptr @global}

; // -----

@0 = global i32 0

; CHECK-LABEL: llvm.func @nameless_global_ref_metadata
; CHECK-SAME: function_metadata
; CHECK-SAME: #llvm.func_metadata<"callees", <#llvm.md_value<@{{mlir\.llvm\.nameless_global_[0-9]+}}>>>
define void @nameless_global_ref_metadata() !callees !0 {
  ret void
}

!0 = !{ptr @0}

; // -----

@ifunc = ifunc void (), ptr @ifunc_resolver
define ptr @ifunc_resolver() {
  ret ptr @ifunc_target
}
define void @ifunc_target() {
  ret void
}

; CHECK-LABEL: llvm.func @ifunc_ref_metadata
; CHECK-SAME: function_metadata
; CHECK-SAME: #llvm.func_metadata<"callees", <#llvm.md_value<@ifunc>>>
define void @ifunc_ref_metadata() !callees !0 {
  ret void
}

!0 = !{ptr @ifunc}
