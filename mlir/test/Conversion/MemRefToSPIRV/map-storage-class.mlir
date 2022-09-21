// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -map-memref-spirv-storage-class='client-api=vulkan' -verify-diagnostics %s -o - | FileCheck %s --check-prefix=VULKAN
// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -map-memref-spirv-storage-class='client-api=opencl' -verify-diagnostics %s -o - | FileCheck %s --check-prefix=OPENCL

// Vulkan Mappings:
//   0 -> StorageBuffer
//   1 -> Generic
//   2 -> [null]
//   3 -> Workgroup
//   4 -> Uniform

// OpenCL Mappings:
//   0 -> CrossWorkgroup
//   1 -> Generic
//   2 -> [null]
//   3 -> Workgroup
//   4 -> UniformConstant

// VULKAN-LABEL: func @operand_result
// OPENCL-LABEL: func @operand_result
func.func @operand_result() {
  // VULKAN: memref<f32, #spv.storage_class<StorageBuffer>>
  // OPENCL: memref<f32, #spv.storage_class<CrossWorkgroup>>
  %0 = "dialect.memref_producer"() : () -> (memref<f32>)
  // VULKAN: memref<4xi32, #spv.storage_class<Generic>>
  // OPENCL: memref<4xi32, #spv.storage_class<Generic>>
  %1 = "dialect.memref_producer"() : () -> (memref<4xi32, 1>)
  // VULKAN: memref<?x4xf16, #spv.storage_class<Workgroup>>
  // OPENCL: memref<?x4xf16, #spv.storage_class<Workgroup>>
  %2 = "dialect.memref_producer"() : () -> (memref<?x4xf16, 3>)
  // VULKAN: memref<*xf16, #spv.storage_class<Uniform>>
  // OPENCL: memref<*xf16, #spv.storage_class<UniformConstant>>
  %3 = "dialect.memref_producer"() : () -> (memref<*xf16, 4>)


  "dialect.memref_consumer"(%0) : (memref<f32>) -> ()
  // VULKAN: memref<4xi32, #spv.storage_class<Generic>>
  // OPENCL: memref<4xi32, #spv.storage_class<Generic>>
  "dialect.memref_consumer"(%1) : (memref<4xi32, 1>) -> ()
  // VULKAN: memref<?x4xf16, #spv.storage_class<Workgroup>>
  // OPENCL: memref<?x4xf16, #spv.storage_class<Workgroup>>
  "dialect.memref_consumer"(%2) : (memref<?x4xf16, 3>) -> ()
  // VULKAN: memref<*xf16, #spv.storage_class<Uniform>>
  // OPENCL: memref<*xf16, #spv.storage_class<UniformConstant>>
  "dialect.memref_consumer"(%3) : (memref<*xf16, 4>) -> ()

  return
}

// -----

// VULKAN-LABEL: func @type_attribute
// OPENCL-LABEL: func @type_attribute
func.func @type_attribute() {
  // VULKAN: attr = memref<i32, #spv.storage_class<Generic>>
  // OPENCL: attr = memref<i32, #spv.storage_class<Generic>>
  "dialect.memref_producer"() { attr = memref<i32, 1> } : () -> ()
  return
}

// -----

// VULKAN-LABEL: func.func @function_io
// OPENCL-LABEL: func.func @function_io
func.func @function_io
  // VULKAN-SAME: (%{{.+}}: memref<f64, #spv.storage_class<Generic>>, %{{.+}}: memref<4xi32, #spv.storage_class<Workgroup>>)
  // OPENCL-SAME: (%{{.+}}: memref<f64, #spv.storage_class<Generic>>, %{{.+}}: memref<4xi32, #spv.storage_class<Workgroup>>)
  (%arg0: memref<f64, 1>, %arg1: memref<4xi32, 3>)
  // VULKAN-SAME: -> (memref<f64, #spv.storage_class<Generic>>, memref<4xi32, #spv.storage_class<Workgroup>>)
  // OPENCL-SAME: -> (memref<f64, #spv.storage_class<Generic>>, memref<4xi32, #spv.storage_class<Workgroup>>)
  -> (memref<f64, 1>, memref<4xi32, 3>) {
  return %arg0, %arg1: memref<f64, 1>, memref<4xi32, 3>
}

// -----

gpu.module @kernel {
// VULKAN-LABEL: gpu.func @function_io
// OPENCL-LABEL: gpu.func @function_io
// VULKAN-SAME: memref<8xi32, #spv.storage_class<StorageBuffer>>
// OPENCL-SAME: memref<8xi32, #spv.storage_class<CrossWorkgroup>>
gpu.func @function_io(%arg0 : memref<8xi32>) kernel { gpu.return }
}

// -----

// VULKAN-LABEL: func.func @region
// OPENCL-LABEL: func.func @region
func.func @region(%cond: i1, %arg0: memref<f32, 1>) {
  scf.if %cond {
    //      VULKAN: "dialect.memref_consumer"(%{{.+}}) {attr = memref<i64, #spv.storage_class<Workgroup>>}
    //      OPENCL: "dialect.memref_consumer"(%{{.+}}) {attr = memref<i64, #spv.storage_class<Workgroup>>}
    // VULKAN-SAME: (memref<f32, #spv.storage_class<Generic>>) -> memref<f32, #spv.storage_class<Generic>>
    // OPENCL-SAME: (memref<f32, #spv.storage_class<Generic>>) -> memref<f32, #spv.storage_class<Generic>>
    %0 = "dialect.memref_consumer"(%arg0) { attr = memref<i64, 3> } : (memref<f32, 1>) -> (memref<f32, 1>)
  }
  return
}

// -----

// VULKAN-LABEL: func @non_memref_types
// OPENCL-LABEL: func @non_memref_types
func.func @non_memref_types(%arg: f32) -> f32 {
  // VULKAN: "dialect.op"(%{{.+}}) {attr = 16 : i64} : (f32) -> f32
  // OPENCL: "dialect.op"(%{{.+}}) {attr = 16 : i64} : (f32) -> f32
  %0 = "dialect.op"(%arg) { attr = 16 } : (f32) -> (f32)
  return %0 : f32
}

// -----

func.func @missing_mapping() {
  // expected-error @+1 {{failed to legalize}}
  %0 = "dialect.memref_producer"() : () -> (memref<f32, 2>)
  return
}

// -----

/// Checks memory maps to OpenCL mapping if Kernel capability is enabled.
module attributes { spv.target_env = #spv.target_env<#spv.vce<v1.0, [Kernel], []>, #spv.resource_limits<>> } {
func.func @operand_result() {
  // CHECK: memref<f32, #spv.storage_class<CrossWorkgroup>>
  %0 = "dialect.memref_producer"() : () -> (memref<f32>)
  // CHECK: memref<4xi32, #spv.storage_class<Generic>>
  %1 = "dialect.memref_producer"() : () -> (memref<4xi32, 1>)
  // CHECK: memref<?x4xf16, #spv.storage_class<Workgroup>>
  %2 = "dialect.memref_producer"() : () -> (memref<?x4xf16, 3>)
  // CHECK: memref<*xf16, #spv.storage_class<UniformConstant>>
  %3 = "dialect.memref_producer"() : () -> (memref<*xf16, 4>)


  "dialect.memref_consumer"(%0) : (memref<f32>) -> ()
  // CHECK: memref<4xi32, #spv.storage_class<Generic>>
  "dialect.memref_consumer"(%1) : (memref<4xi32, 1>) -> ()
  // CHECK: memref<?x4xf16, #spv.storage_class<Workgroup>>
  "dialect.memref_consumer"(%2) : (memref<?x4xf16, 3>) -> ()
  // CHECK: memref<*xf16, #spv.storage_class<UniformConstant>>
  "dialect.memref_consumer"(%3) : (memref<*xf16, 4>) -> ()

  return
}
}

// -----

/// Checks memory maps to Vulkan mapping if Shader capability is enabled.
module attributes { spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], []>, #spv.resource_limits<>> } {
func.func @operand_result() {
  // CHECK: memref<f32, #spv.storage_class<StorageBuffer>>
  %0 = "dialect.memref_producer"() : () -> (memref<f32>)
  // CHECK: memref<4xi32, #spv.storage_class<Generic>>
  %1 = "dialect.memref_producer"() : () -> (memref<4xi32, 1>)
  // CHECK: memref<?x4xf16, #spv.storage_class<Workgroup>>
  %2 = "dialect.memref_producer"() : () -> (memref<?x4xf16, 3>)
  // CHECK: memref<*xf16, #spv.storage_class<Uniform>>
  %3 = "dialect.memref_producer"() : () -> (memref<*xf16, 4>)


  "dialect.memref_consumer"(%0) : (memref<f32>) -> ()
  // CHECK: memref<4xi32, #spv.storage_class<Generic>>
  "dialect.memref_consumer"(%1) : (memref<4xi32, 1>) -> ()
  // CHECK: memref<?x4xf16, #spv.storage_class<Workgroup>>
  "dialect.memref_consumer"(%2) : (memref<?x4xf16, 3>) -> ()
  // CHECK: memref<*xf16, #spv.storage_class<Uniform>>
  "dialect.memref_consumer"(%3) : (memref<*xf16, 4>) -> ()
  return
}
}