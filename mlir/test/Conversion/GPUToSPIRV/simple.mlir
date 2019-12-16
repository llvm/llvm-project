// RUN: mlir-opt -pass-pipeline='convert-gpu-to-spirv{workgroup-size=32,4}' %s -o - | FileCheck %s

module attributes {gpu.container_module} {

  module @kernels attributes {gpu.kernel_module} {
    // CHECK:       spv.module "Logical" "GLSL450" {
    // CHECK-LABEL: func @kernel_1
    // CHECK-SAME: {{%.*}}: f32 {spirv.interface_var_abi = {binding = 0 : i32, descriptor_set = 0 : i32, storage_class = 12 : i32{{[}][}]}}
    // CHECK-SAME: {{%.*}}: !spv.ptr<!spv.struct<!spv.array<12 x f32 [4]> [0]>, StorageBuffer> {spirv.interface_var_abi = {binding = 1 : i32, descriptor_set = 0 : i32, storage_class = 12 : i32{{[}][}]}}
    // CHECK-SAME: spirv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}
    gpu.func @kernel_1(%arg0 : f32, %arg1 : memref<12xf32>) attributes {gpu.kernel} {
      // CHECK: spv.Return
      gpu.return
    }
    // CHECK: attributes {capabilities = ["Shader"], extensions = ["SPV_KHR_storage_buffer_storage_class"]}
  }

  func @foo() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32>)
    %cst = constant 1 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index, f32, memref<12xf32>) -> ()
    return
  }
}
