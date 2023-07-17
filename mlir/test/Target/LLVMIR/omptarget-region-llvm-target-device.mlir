// This test checks that the name of the generated kernel function is using the
// name stored in the omp.outline_parent_name attribute.
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  llvm.func @writeindex_omp_outline_0_(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>) attributes {omp.outline_parent_name = "writeindex_"} {
    omp.target   map((from -> %arg0 : !llvm.ptr<i32>), (implicit -> %arg1: !llvm.ptr<i32>)) {
      %0 = llvm.mlir.constant(20 : i32) : i32
      %1 = llvm.mlir.constant(10 : i32) : i32
      llvm.store %1, %arg0 : !llvm.ptr<i32>
      llvm.store %0, %arg1 : !llvm.ptr<i32>
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define {{.*}} void @__omp_offloading_{{.*}}_{{.*}}_writeindex__l7(ptr {{.*}}, ptr {{.*}}) {
