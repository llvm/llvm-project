// RUN: mlir-opt -split-input-file -verify-diagnostics %s

llvm.func @malloc(i64) -> !llvm.ptr
func.func @func2(%arg0: index, %arg1: memref<13x13xi64>, %arg2: index) {
  %cst_7 = arith.constant dense<1526248407> : vector<1xi64>
  %1 = llvm.mlir.constant(1 : index) : i64
  %101 = vector.insert %1, %cst_7 [0] : i64 into vector<1xi64>
  vector.print %101 : vector<1xi64>
  return
}
