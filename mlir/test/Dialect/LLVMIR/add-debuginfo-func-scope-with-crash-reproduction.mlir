// Test that the enable-debug-info-scope-on-llvm-func pass can create its
// distinct attributes when running in the crash reproducer thread.

// RUN: mlir-opt --mlir-disable-threading --mlir-pass-pipeline-crash-reproducer=. \
// RUN:          --pass-pipeline="builtin.module(ensure-debug-info-scope-on-llvm-func)" \
// RUN:          --mlir-print-debuginfo %s | FileCheck %s

// RUN: mlir-opt --mlir-pass-pipeline-crash-reproducer=. \
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
// CHECK: #di_compile_unit = #llvm.di_compile_unit<id = distinct[{{.*}}]<>,
// CHECK: #di_subprogram = #llvm.di_subprogram<id = distinct[{{.*}}]<>
