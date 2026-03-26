// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify that host offloading doesn't crash the OMPIRBuilder.
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_target_device = true} {
  llvm.func @omp_target_region_host_device() {
    omp.target {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK:      define void @omp_target_region_host_device()
// CHECK:      define weak_odr protected void @__omp_offloading_{{[^_]+}}_{{[^_]+}}_omp_target_region_host_device_l{{[0-9]+}}(ptr %[[ADDR_A:.*]])
