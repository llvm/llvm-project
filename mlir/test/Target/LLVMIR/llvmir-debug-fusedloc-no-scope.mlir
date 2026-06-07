// RUN: mlir-translate -mlir-to-llvmir --split-input-file %s | FileCheck %s

// Verify that a FusedLoc without debug scope metadata does not crash when
// sub-locations translate to nullptr (no scope available). This pattern is
// produced by ClangIR which wraps source locations in FusedLoc without
// attaching a DILocalScopeAttr.

// CHECK-LABEL: define void @fusedloc_no_scope_no_debug()
// CHECK-NOT: !dbg
// CHECK: }
llvm.func @fusedloc_no_scope_no_debug() {
  llvm.return loc(fused["test.c":1:1, "test.c":2:2])
}

// -----

// Verify the same situation when the function has a subprogram but the
// FusedLoc contains UnknownLoc sub-locations (which always translate to
// nullptr regardless of scope).

#di_file = #llvm.di_file<"test.c" in "/tmp">
#di_cu = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C,
  file = #di_file, isOptimized = false, emissionKind = Full
>
#void_return = #llvm.di_null_type
#void_type = #llvm.di_subroutine_type<types = #void_return>
#di_sp = #llvm.di_subprogram<
  id = distinct[1]<>, compileUnit = #di_cu,
  scope = #di_file, name = "fusedloc_unknown_sublocs",
  file = #di_file, subprogramFlags = Definition, type = #void_type
>

// CHECK-LABEL: define void @fusedloc_unknown_sublocs() !dbg
// CHECK-NOT: !dbg
// CHECK: }
llvm.func @fusedloc_unknown_sublocs() {
  llvm.return loc(fused[unknown, unknown])
} loc(fused<#di_sp>["test.c":1:1])
