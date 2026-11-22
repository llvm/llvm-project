// RUN: mlir-opt %s --verify-roundtrip --mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: llvm.func @memprof_attrs
// CHECK: llvm.call
// CHECK-SAME: llvm.callsite = array<i64: 123456789>
// CHECK-SAME: llvm.memprof = [#llvm.memprof_mib<call_stack = [123456789], alloc_type = "cold", context_sizes = <id = 999, size = 4096>>]

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @memprof_attrs() -> !llvm.ptr {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.call @alloc(%0) {
    llvm.callsite = array<i64: 123456789>,
    llvm.memprof = [#llvm.memprof_mib<call_stack = [123456789], alloc_type = "cold",
                                        context_sizes = <id = 999, size = 4096>>]
  } : (i64) -> !llvm.ptr
  llvm.return %1 : !llvm.ptr
}
