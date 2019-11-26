// RUN: mlir-opt -spirv-lower-abi-attrs -verify-diagnostics %s -o - | FileCheck %s

// CHECK-LABEL: spv.module
spv.module "Logical" "GLSL450" {
  // CHECK-DAG:    spv.globalVariable [[VAR0:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<f32 [0]>, StorageBuffer>
  // CHECK-DAG:    spv.globalVariable [[VAR1:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x f32 [4]> [0]>, StorageBuffer>
  // CHECK:    func [[FN:@.*]]()
  func @kernel_1(%arg0: f32
                {spirv.interface_var_abi = {binding = 0 : i32,
                                            descriptor_set = 0 : i32,
                                            storage_class = 12 : i32}},
                 %arg1: !spv.ptr<!spv.struct<!spv.array<12 x f32>>, StorageBuffer>
                 {spirv.interface_var_abi = {binding = 1 : i32,
                                             descriptor_set = 0 : i32,
                                             storage_class = 12 : i32}})
  attributes  {spirv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    // CHECK: [[ARG1:%.*]] = spv._address_of [[VAR1]]
    // CHECK: [[ADDRESSARG0:%.*]] = spv._address_of [[VAR0]]
    // CHECK: [[CONST0:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0PTR:%.*]] = spv.AccessChain [[ADDRESSARG0]]{{\[}}[[CONST0]]
    // CHECK: [[ARG0:%.*]] = spv.Load "StorageBuffer" [[ARG0PTR]]
    // CHECK: spv.Return
    spv.Return
  }
  // CHECK: spv.EntryPoint "GLCompute" [[FN]]
  // CHECK: spv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
} attributes {capabilities = ["Shader"], extensions = ["SPV_KHR_storage_buffer_storage_class"]}
