// RUN: mlir-opt -int-range-optimizations %s 

%0 = scf.execute_region -> tensor<5x16xi16> {
  llvm.unreachable
}
