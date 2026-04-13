// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true,  omp.is_gpu = true, omp.target_triples = ["spirv64-intel"], llvm.target_triple = "spirv64-intel"} {
// CHECK: call spir_func i32 @__kmpc_target_init
// CHECK: call spir_func void @__kmpc_target_deinit
  llvm.func @target_if_variable(%x : i1) {
    omp.target if(%x) {
      omp.terminator
    }
    llvm.return
  }
 }
