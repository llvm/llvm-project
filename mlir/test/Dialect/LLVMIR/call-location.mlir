// RUN: mlir-opt %s -split-input-file | FileCheck %s

#di_file = #llvm.di_file<"file.cpp" in "/folder/">
#di_compile_unit = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file,
  isOptimized = true, emissionKind = Full
>
#di_subprogram = #llvm.di_subprogram<
  compileUnit = #di_compile_unit, scope = #di_file,
  name = "missing_debug_loc", file = #di_file,
  subprogramFlags = "Definition|Optimized"
>
#di_subprogram1 = #llvm.di_subprogram<
  compileUnit = #di_compile_unit, scope = #di_file,
  name = "invalid_call_debug_locs", file = #di_file,
  subprogramFlags = "Definition|Optimized"
>
#loc = loc(unknown)
#loc1 = loc("file.cpp":24:0)
#loc2 = loc(fused<#di_subprogram>[#loc1])
#loc3 = loc("file.cpp":42:0)
#loc4 = loc(fused<#di_subprogram1>[#loc3])

llvm.func @external_func() loc(#loc2)

llvm.func @call_debug_locs() {
  llvm.call @external_func() : () -> () loc(#loc)
  llvm.return
} loc(#loc4)

// CHECK: llvm.call @external_func()
