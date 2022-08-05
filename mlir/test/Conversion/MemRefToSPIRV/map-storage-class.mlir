// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -map-memref-spirv-storage-class='client-api=vulkan' -verify-diagnostics %s -o - | FileCheck %s --check-prefix=VULKAN

// Vulkan Mappings:
//   0 -> StorageBuffer (12)
//   1 -> Generic (8)
//   3 -> Workgroup (4)
//   4 -> Uniform (2)
// TODO: create a StorageClass wrapper class so we can print the symbolc
// storage class (instead of the backing IntegerAttr) and be able to
// round trip the IR.

// VULKAN-LABEL: func @operand_result
func.func @operand_result() {
  // VULKAN: memref<f32, 12 : i32>
  %0 = "dialect.memref_producer"() : () -> (memref<f32>)
  // VULKAN: memref<4xi32, 8 : i32>
  %1 = "dialect.memref_producer"() : () -> (memref<4xi32, 1>)
  // VULKAN: memref<?x4xf16, 4 : i32>
  %2 = "dialect.memref_producer"() : () -> (memref<?x4xf16, 3>)
  // VULKAN: memref<*xf16, 2 : i32>
  %3 = "dialect.memref_producer"() : () -> (memref<*xf16, 4>)


  "dialect.memref_consumer"(%0) : (memref<f32>) -> ()
  // VULKAN: memref<4xi32, 8 : i32>
  "dialect.memref_consumer"(%1) : (memref<4xi32, 1>) -> ()
  // VULKAN: memref<?x4xf16, 4 : i32>
  "dialect.memref_consumer"(%2) : (memref<?x4xf16, 3>) -> ()
  // VULKAN: memref<*xf16, 2 : i32>
  "dialect.memref_consumer"(%3) : (memref<*xf16, 4>) -> ()

  return
}

// -----

// VULKAN-LABEL: func @type_attribute
func.func @type_attribute() {
  // VULKAN: attr = memref<i32, 8 : i32>
  "dialect.memref_producer"() { attr = memref<i32, 1> } : () -> ()
  return
}

// -----

// VULKAN-LABEL: func @function_io
func.func @function_io
  // VULKAN-SAME: (%{{.+}}: memref<f64, 8 : i32>, %{{.+}}: memref<4xi32, 4 : i32>)
  (%arg0: memref<f64, 1>, %arg1: memref<4xi32, 3>)
  // VULKAN-SAME: -> (memref<f64, 8 : i32>, memref<4xi32, 4 : i32>)
  -> (memref<f64, 1>, memref<4xi32, 3>) {
  return %arg0, %arg1: memref<f64, 1>, memref<4xi32, 3>
}

// -----

// VULKAN: func @region
func.func @region(%cond: i1, %arg0: memref<f32, 1>) {
  scf.if %cond {
    //      VULKAN: "dialect.memref_consumer"(%{{.+}}) {attr = memref<i64, 4 : i32>}
    // VULKAN-SAME: (memref<f32, 8 : i32>) -> memref<f32, 8 : i32>
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
