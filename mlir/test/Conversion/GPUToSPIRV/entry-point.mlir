// RUN: mlir-opt -test-spirv-entry-point-abi %s | FileCheck %s -check-prefix=DEFAULT
// RUN: mlir-opt -test-spirv-entry-point-abi="workgroup-size=32" %s | FileCheck %s -check-prefix=WG32
// RUN: mlir-opt -test-spirv-entry-point-abi="subgroup-size=4" %s | FileCheck %s -check-prefix=SG4
// RUN: mlir-opt -test-spirv-entry-point-abi="target-width=32" %s | FileCheck %s -check-prefix=TW32
// RUN: mlir-opt -test-spirv-entry-point-abi="workgroup-size=32,8 subgroup-size=4 target-width=32" %s | FileCheck %s -check-prefix=WG32_8-SG4-TW32

//      DEFAULT: gpu.func @foo()
// DEFAULT-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>

//      WG32: gpu.func @foo()
// WG32-SAME:  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>

//      SG4: gpu.func @foo()
// SG4-SAME:  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1], subgroup_size = 4>

//      TW32: gpu.func @foo()
// TW32-SAME:  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1], target_width = 32>

//      WG32_8-SG4-TW32: gpu.func @foo()
// WG32_8-SG4-TW32-SAME:  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 8, 1], subgroup_size = 4, target_width = 32>

gpu.module @kernels {
  gpu.func @foo() kernel {
    gpu.return
  }
}
