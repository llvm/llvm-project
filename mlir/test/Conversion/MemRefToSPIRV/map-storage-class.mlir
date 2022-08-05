// RUN: mlir-opt -split-input-file -allow-unregistered-dialect -map-memref-spirv-storage-class='mappings=0=StorageBuffer,1=Uniform,2=Workgroup,3=PushConstant' -verify-diagnostics %s -o - | FileCheck %s

// Mappings:
//   0 -> StorageBuffer (12)
//   2 -> Workgroup (4)
//   1 -> Uniform (2)
//   3 -> PushConstant (9)
// TODO: create a StorageClass wrapper class so we can print the symbolc
// storage class (instead of the backing IntegerAttr) and be able to
// round trip the IR.

// CHECK-LABEL: func @operand_result
func.func @operand_result() {
  // CHECK: memref<f32, 12 : i32>
  %0 = "dialect.memref_producer"() : () -> (memref<f32>)
  // CHECK: memref<4xi32, 2 : i32>
  %1 = "dialect.memref_producer"() : () -> (memref<4xi32, 1>)
  // CHECK: memref<?x4xf16, 4 : i32>
  %2 = "dialect.memref_producer"() : () -> (memref<?x4xf16, 2>)
  // CHECK: memref<*xf16, 9 : i32>
  %3 = "dialect.memref_producer"() : () -> (memref<*xf16, 3>)


  "dialect.memref_consumer"(%0) : (memref<f32>) -> ()
  // CHECK: memref<4xi32, 2 : i32>
  "dialect.memref_consumer"(%1) : (memref<4xi32, 1>) -> ()
  // CHECK: memref<?x4xf16, 4 : i32>
  "dialect.memref_consumer"(%2) : (memref<?x4xf16, 2>) -> ()
  // CHECK: memref<*xf16, 9 : i32>
  "dialect.memref_consumer"(%3) : (memref<*xf16, 3>) -> ()

  return
}

// -----

// CHECK-LABEL: func @type_attribute
func.func @type_attribute() {
  // CHECK: attr = memref<i32, 2 : i32>
  "dialect.memref_producer"() { attr = memref<i32, 1> } : () -> ()
  return
}

// -----

// CHECK-LABEL: func @function_io
func.func @function_io
  // CHECK-SAME: (%{{.+}}: memref<f64, 4 : i32>, %{{.+}}: memref<4xi32, 9 : i32>)
  (%arg0: memref<f64, 2>, %arg1: memref<4xi32, 3>)
  // CHECK-SAME: -> (memref<f64, 4 : i32>, memref<4xi32, 9 : i32>)
  -> (memref<f64, 2>, memref<4xi32, 3>) {
  return %arg0, %arg1: memref<f64, 2>, memref<4xi32, 3>
}

// -----

// CHECK: func @region
func.func @region(%cond: i1, %arg0: memref<f32, 1>) {
  scf.if %cond {
    //      CHECK: "dialect.memref_consumer"(%{{.+}}) {attr = memref<i64, 4 : i32>}
    // CHECK-SAME: (memref<f32, 2 : i32>) -> memref<f32, 2 : i32>
    %0 = "dialect.memref_consumer"(%arg0) { attr = memref<i64, 2> } : (memref<f32, 1>) -> (memref<f32, 1>)
  }
  return
}

// -----

// CHECK-LABEL: func @non_memref_types
func.func @non_memref_types(%arg: f32) -> f32 {
  // CHECK: "dialect.op"(%{{.+}}) {attr = 16 : i64} : (f32) -> f32
  %0 = "dialect.op"(%arg) { attr = 16 } : (f32) -> (f32)
  return %0 : f32
}

// -----

func.func @missing_mapping() {
  // expected-error @+1 {{failed to legalize}}
  %0 = "dialect.memref_producer"() : () -> (memref<f32, 128>)
  return
}
