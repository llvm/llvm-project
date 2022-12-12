// RUN: mlir-opt -test-spirv-entry-point-abi %s | FileCheck %s -check-prefix=DEFAULT
// RUN: mlir-opt -test-spirv-entry-point-abi="workgroup-size=32" %s | FileCheck %s -check-prefix=WG32

//      DEFAULT: gpu.func @foo()
// DEFAULT-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>

//      WG32: gpu.func @foo()
// WG32-SAME:  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>

gpu.module @kernels {
  gpu.func @foo() kernel {
    gpu.return
  }
}
