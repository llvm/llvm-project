// RUN: mlir-runner %s -e main -entry-point-result=i64 2>&1 | \
// RUN: FileCheck %s --implicit-check-not="JIT session error"
// RUN: mlir-runner %s -e main -entry-point-result=i64 \
// RUN:   --enable-gdb-listener=false --enable-perf-listener=false 2>&1 | \
// RUN: FileCheck %s --implicit-check-not="JIT session error"
// REQUIRES: host-supports-jit
// XFAIL: system-aix

// Exercise address materialization for JIT-allocated data. In particular,
// RISC-V JIT code must not use absolute medlow addressing for this global.

// CHECK: 42
module {
  llvm.mlir.global internal @value(42 : i64) : i64
  llvm.func @main() -> i64 {
    %addr = llvm.mlir.addressof @value : !llvm.ptr
    %value = llvm.load %addr : !llvm.ptr -> i64
    llvm.return %value : i64
  }
}
