// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa-mark-graph-constants %s | FileCheck %s

// CHECK-LABEL: func.func @large_const
func.func @large_const() -> tensor<17xi32> {
  // CHECK: "tosa.const"() <{values = {{.*}}> {grapharm.graph_constant_id = 0 : i32} : () -> tensor<17xi32>
  %res = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<17xi32>}> : () -> tensor<17xi32>
  return %res : tensor<17xi32>
}

// -----

// CHECK-LABEL: func.func @small_const
// CHECK-NOT: grapharm.graph_constant_id
func.func @small_const() -> tensor<16xi32> {
  %res = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>}> : () -> tensor<16xi32>
  return %res : tensor<16xi32>
}

// -----

// CHECK-LABEL: func.func @large_const_shape
func.func @large_const_shape() -> !tosa.shape<33> {
  // CHECK: tosa.const_shape {grapharm.graph_constant_id = 0 : i32, values = {{.*}}} : () -> !tosa.shape<33>
  %res = "tosa.const_shape"() <{values = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]> : tensor<33xindex>}> : () -> !tosa.shape<33>
  return %res : !tosa.shape<33>
}

// -----

// CHECK-LABEL: func.func @small_const_shape
// CHECK-NOT: grapharm.graph_constant_id
func.func @small_const_shape() -> !tosa.shape<32> {
  %res = "tosa.const_shape"() <{values = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]> : tensor<32xindex>}> : () -> !tosa.shape<32>
  return %res : !tosa.shape<32>
}

// -----

// CHECK-LABEL: func.func @mixed_large_constants
func.func @mixed_large_constants() -> (tensor<17xi32>, tensor<18xi32>, !tosa.shape<33>) {
  // CHECK: "tosa.const"() <{values = {{.*}}> {grapharm.graph_constant_id = 0 : i32} : () -> tensor<17xi32>
  %const0 = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<17xi32>}> : () -> tensor<17xi32>
  // CHECK: "tosa.const"() <{values = {{.*}}> {grapharm.graph_constant_id = 1 : i32} : () -> tensor<18xi32>
  %const1 = "tosa.const"() <{values = dense<[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]> : tensor<18xi32>}> : () -> tensor<18xi32>
  // CHECK: tosa.const_shape {grapharm.graph_constant_id = 2 : i32, values = {{.*}}} : () -> !tosa.shape<33>
  %shape = "tosa.const_shape"() <{values = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]> : tensor<33xindex>}> : () -> !tosa.shape<33>
  return %const0, %const1, %shape : tensor<17xi32>, tensor<18xi32>, !tosa.shape<33>
}
