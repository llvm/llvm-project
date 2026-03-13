// Test that the target_features and target_cpu llvm.func attributes are
// forwarded to outlined target region functions.

// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  llvm.func @omp_target_region() attributes {
    target_cpu = "x86-64",
    target_features = #llvm.target_features<["+mmx", "+sse"]>
  } {
    omp.target {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define void @omp_target_region() #[[ATTRS:.*]] {
// CHECK: define internal void @__omp_offloading_{{.*}}_omp_target_region_{{.*}}() #[[ATTRS]] {

// CHECK: attributes #[[ATTRS]] = {
// CHECK-SAME: "target-cpu"="x86-64"
// CHECK-SAME: "target-features"="+mmx,+sse"
