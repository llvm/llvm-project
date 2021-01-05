// RUN: mlir-opt -test-spirv-entry-point-abi %s | FileCheck %s -check-prefix=DEFAULT
// RUN: mlir-opt -test-spirv-entry-point-abi="workgroup-size=32" %s | FileCheck %s -check-prefix=WG32

//      DEFAULT: gpu.func @foo()
// DEFAULT-SAME: spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}

//      WG32: gpu.func @foo()
// WG32-SAME:  spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}

gpu.module @kernels {
  gpu.func @foo() kernel {
    gpu.return
  }
}
