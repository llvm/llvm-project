// RUN: mlir-translate -mlir-to-llvmir --split-input-file %s | FileCheck %s

// Verify that ArtificialLoc translates to DILocation(line: 0, column: 0).
// The DWARF specification allows the compiler to use the special line number 0
// to indicate code that cannot be attributed to any source location.

// When a valid scope is available, ArtificialLoc produces a DILocation with
// line=0 attached to the instruction, preserving the enclosing scope.
//
// CHECK-LABEL: define void @func_artificial_in_debug_scope
// CHECK-SAME:  !dbg ![[SP:[0-9]+]]
// CHECK:       call void @callee(), !dbg ![[ARTLOC:[0-9]+]]
// CHECK-DAG:   ![[ARTLOC]] = !DILocation(line: 0, scope: ![[SP]])

#file = #llvm.di_file<"test.mlir" in "/test/">
#cu = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #file,
  producer = "MLIR", isOptimized = false, emissionKind = Full>
#spTy = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp = #llvm.di_subprogram<
  id = distinct[1]<>, compileUnit = #cu, scope = #file,
  name = "func_artificial_in_debug_scope", file = #file,
  subprogramFlags = "Definition", type = #spTy>

llvm.func @callee() {
  llvm.return
}

llvm.func @func_artificial_in_debug_scope() {
  llvm.call @callee() : () -> () loc(artificial)
  llvm.return
} loc(fused<#sp>["test.mlir":1:1])

// -----

// When no enclosing scope exists, ArtificialLoc produces no !dbg metadata,
// consistent with the behaviour of loc(unknown).
//
// CHECK-LABEL: define void @func_artificial_no_scope()
// CHECK-NOT:   !dbg

llvm.func @func_artificial_no_scope() {
  llvm.return loc(artificial)
} loc(artificial)
