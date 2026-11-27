// RUN: mlir-runner %s -e main -entry-point-result=i64 | FileCheck %s

// XFAIL: system-aix

// A module carrying the data layout specification that the LLVM IR importer
// and the llvm-target-to-data-layout pass attach parses and runs.
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>>} {
  // CHECK: 42
  llvm.func @main() -> i64 {
    %0 = llvm.mlir.constant(42 : i64) : i64
    llvm.return %0 : i64
  }
}
