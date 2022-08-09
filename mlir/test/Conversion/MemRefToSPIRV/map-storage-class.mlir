// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -map-memref-spirv-storage-class='client-api=vulkan' -verify-diagnostics %s -o - | FileCheck %s --check-prefix=VULKAN

// Vulkan Mappings:
//   0 -> StorageBuffer
//   1 -> Generic
//   2 -> [null]
//   3 -> Workgroup
//   4 -> Uniform

// VULKAN-LABEL: func @operand_result
func.func @operand_result() {
  // VULKAN: memref<f32, #spv.storage_class<StorageBuffer>>
  %0 = "dialect.memref_producer"() : () -> (memref<f32>)
  // VULKAN: memref<4xi32, #spv.storage_class<Generic>>
  %1 = "dialect.memref_producer"() : () -> (memref<4xi32, 1>)
  // VULKAN: memref<?x4xf16, #spv.storage_class<Workgroup>>
  %2 = "dialect.memref_producer"() : () -> (memref<?x4xf16, 3>)
  // VULKAN: memref<*xf16, #spv.storage_class<Uniform>>
  %3 = "dialect.memref_producer"() : () -> (memref<*xf16, 4>)


  "dialect.memref_consumer"(%0) : (memref<f32>) -> ()
  // VULKAN: memref<4xi32, #spv.storage_class<Generic>>
  "dialect.memref_consumer"(%1) : (memref<4xi32, 1>) -> ()
  // VULKAN: memref<?x4xf16, #spv.storage_class<Workgroup>>
  "dialect.memref_consumer"(%2) : (memref<?x4xf16, 3>) -> ()
  // VULKAN: memref<*xf16, #spv.storage_class<Uniform>>
  "dialect.memref_consumer"(%3) : (memref<*xf16, 4>) -> ()

  return
}

// -----

// VULKAN-LABEL: func @type_attribute
func.func @type_attribute() {
  // VULKAN: attr = memref<i32, #spv.storage_class<Generic>>
  "dialect.memref_producer"() { attr = memref<i32, 1> } : () -> ()
  return
}

// -----

// VULKAN-LABEL: func @function_io
func.func @function_io
  // VULKAN-SAME: (%{{.+}}: memref<f64, #spv.storage_class<Generic>>, %{{.+}}: memref<4xi32, #spv.storage_class<Workgroup>>)
  (%arg0: memref<f64, 1>, %arg1: memref<4xi32, 3>)
  // VULKAN-SAME: -> (memref<f64, #spv.storage_class<Generic>>, memref<4xi32, #spv.storage_class<Workgroup>>)
  -> (memref<f64, 1>, memref<4xi32, 3>) {
  return %arg0, %arg1: memref<f64, 1>, memref<4xi32, 3>
}

// -----

// VULKAN: func @region
func.func @region(%cond: i1, %arg0: memref<f32, 1>) {
  scf.if %cond {
    //      VULKAN: "dialect.memref_consumer"(%{{.+}}) {attr = memref<i64, #spv.storage_class<Workgroup>>}
    // VULKAN-SAME: (memref<f32, #spv.storage_class<Generic>>) -> memref<f32, #spv.storage_class<Generic>>
    %0 = "dialect.memref_consumer"(%arg0) { attr = memref<i64, 3> } : (memref<f32, 1>) -> (memref<f32, 1>)
  }
  return
}

// -----

// VULKAN-LABEL: func @non_memref_types
func.func @non_memref_types(%arg: f32) -> f32 {
  // VULKAN: "dialect.op"(%{{.+}}) {attr = 16 : i64} : (f32) -> f32
  %0 = "dialect.op"(%arg) { attr = 16 } : (f32) -> (f32)
  return %0 : f32
}

// -----

func.func @missing_mapping() {
  // expected-error @+1 {{failed to legalize}}
  %0 = "dialect.memref_producer"() : () -> (memref<f32, 2>)
  return
}
