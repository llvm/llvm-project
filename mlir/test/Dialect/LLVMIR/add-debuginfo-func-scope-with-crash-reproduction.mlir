// Test that the enable-debug-info-scope-on-llvm-func pass can create its
// DI attributes when running in the crash reproducer thread,

// RUN: mlir-opt --mlir-disable-threading --mlir-pass-pipeline-crash-reproducer=. \
// RUN:          --pass-pipeline="builtin.module(ensure-debug-info-scope-on-llvm-func)" \
// RUN:          --mlir-print-debuginfo %s | FileCheck %s

module {
  llvm.func @func_no_debug() {
    llvm.return loc(unknown)
  } loc(unknown)
} loc(unknown)

// CHECK-LABEL: llvm.func @func_no_debug()
// CHECK: llvm.return loc(#loc
// CHECK: loc(#loc[[LOC:[0-9]+]])
// CHECK: #di_file = #llvm.di_file<"<unknown>" in "">
// CHECK: #di_subprogram = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "func_no_debug", linkageName = "func_no_debug", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
// CHECK: #loc[[LOC]] = loc(fused<#di_subprogram>
