// RUN: tr-opt %s --tr-annotate-gpu-target | FileCheck %s

// Milestone 16: GPUTargetInfo is recorded on the module. Warp size and
// the 256-thread / 8-warp launch are the baseline schedule. SM / register
// / shared-memory numbers are target properties, not source semantics.

// CHECK: module attributes
// CHECK-DAG: tr.target.warp_size = 32
// CHECK-DAG: tr.target.threads_per_block = 256
// CHECK-DAG: tr.target.warps_per_block = 8
// CHECK-DAG: tr.target.num_sms = 108
// CHECK-DAG: tr.target.max_threads_per_block = 1024
// CHECK-DAG: tr.target.max_warps_per_sm = 64
// CHECK-DAG: tr.target.max_blocks_per_sm = 32
// CHECK-DAG: tr.target.registers_per_sm = 65536
// CHECK-DAG: tr.target.max_registers_per_thread = 255
// CHECK-DAG: tr.target.shared_memory_per_sm = 167936
// CHECK-DAG: tr.target.shared_memory_per_block = 166912
// CHECK-DAG: tr.target.memory_bandwidth_gbs = 1.555000e+03
// CHECK-DAG: tr.target.fp32_peak_tflops = 1.950000e+01

module {
  func.func @dummy() {
    return
  }
}
