// RUN: mlir-opt -xegpu-subgroup-distribute='print-analysis-only=true' -split-input-file %s | FileCheck %s

// CHECK: function: test_dpas_f16:
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load_nd %[[T1]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.dpas %[[T2]], %[[T3]], %{{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %{{.*}} = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_dpas_f16(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}


// -----
// CHECK: function: test_dpas_i8:
// CHECK-NEXT: argument: <block argument> of type 'vector<8x32xi8>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 2]
// CHECK-NEXT: argument: <block argument> of type 'vector<32x16xi8>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [4, 1]
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xi32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.dpas %{{.*}} : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_dpas_i8(%arg0: vector<8x32xi8>, %arg1: vector<32x16xi8>, %arg2: memref<8x16xi32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.dpas %arg0, %arg1 : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
  %1 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
  xegpu.store_nd %0, %1  : vector<8x16xi32>, !xegpu.tensor_desc<8x16xi32>
  return
}

// -----
// CHECK: function: test_load_with_transpose_effect:
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16, 1], wi_data: [1, 2]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load_nd %[[T1]] <{transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.dpas %[[T2]], %[[T3]], %[[CST]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T5:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_load_with_transpose_effect(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 <{transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %2, %3, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK: function: test_vector_transpose:
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16, 1], wi_data: [1, 2]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load_nd %[[T1]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16, 1], wi_data: [1, 2]
// CHECK-NEXT: op    : %[[T4:.*]] = vector.transpose %[[T3]], [1, 0] : vector<16x16xf16> to vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T5:.*]] = xegpu.dpas %[[T2]], %[[T4]], %[[CST]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T6:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_vector_transpose(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = vector.transpose %3, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %5 = xegpu.dpas %2, %4, %cst : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %6 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %5, %6  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK: function: test_extf_truncf:
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = arith.extf %[[T1]] : vector<16x16xf16> to vector<16x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = arith.truncf %[[T2]] : vector<16x16xf32> to vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.dpas %[[T0]], %[[T3]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: Not assigned.
func.func @test_extf_truncf(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.extf %1 : vector<16x16xf16> to vector<16x16xf32>
  %3 = arith.truncf %2 : vector<16x16xf32> to vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----
// CHECK: function: test_load_gather_with_transpose_effect:
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<256xf16>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[CST0:.*]] = arith.constant dense<true> : vector<16xi1>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.create_tdesc %{{.*}}, %[[CST]] : memref<256xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16, 1], wi_data: [1, 2]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load %[[T2]], %[[CST0]] <{transpose}> : !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>>, vector<16xi1> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.dpas %[[T1]], %[[T3]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T5:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_load_gather_with_transpose_effect(%arg0: memref<8x16xf16>, %arg1: memref<256xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %cst = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %2 = xegpu.create_tdesc %arg1, %cst : memref<256xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>>
  %3 = xegpu.load %2, %cst_0 <{transpose}> : !xegpu.tensor_desc<16x16xf16, #xegpu.scatter_tdesc_attr<chunk_size = 16 : i64>>, vector<16xi1> -> vector<16x16xf16>
  %4 = xegpu.dpas %1, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK: function: test_load_gather_1d:
// CHECK: argument: <block argument> of type 'memref<256xf32>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16xf32>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[CST0:.*]] = arith.constant dense<true> : vector<16xi1>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_tdesc %{{.*}}, %[[CST]] : memref<256xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[T1]] = xegpu.load %[[T0]], %[[CST0]]  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
func.func @test_load_gather_1d(%arg0: memref<256xf32>, %arg1: !xegpu.tensor_desc<16xf32>) {
  %cst = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %0 = xegpu.create_tdesc %arg0, %cst : memref<256xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  %1 = xegpu.load %0, %cst_0  : !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1> -> vector<16xf32>
  xegpu.store_nd %1, %arg1  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
// CHECK: function: test_store_scatter_with_transpose_effect:
// CHECK-NEXT: argument: <block argument> of type 'memref<128xf32>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[CST0:.*]] = arith.constant dense<true> : vector<16xi1>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[CST1:.*]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_tdesc %{{.*}}, %[[CST1]] : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16, 1], wi_data: [1, 1]
func.func @test_store_scatter_with_transpose_effect(%arg0: memref<128xf32>) {
  %cst = arith.constant dense<1.000000e+00> : vector<8x16xf32>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %cst_1 = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %0 = xegpu.create_tdesc %arg0, %cst_1 : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>
  xegpu.store %cst, %0, %cst_0 <{transpose}> : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<chunk_size = 8 : i64>>, vector<16xi1>
  return
}

// -----
// CHECK: function: test_store_scatter_1d:
// CHECK-NEXT: argument: <block argument> of type 'vector<16xf32>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [16], wi_data: [1]
// CHECK-NEXT: argument: <block argument> of type 'memref<256xf32>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[CST1:.*]] = arith.constant dense<true> : vector<16xi1>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_tdesc %{{.*}}, %[[CST]] : memref<256xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
func.func @test_store_scatter_1d(%arg0: vector<16xf32>, %arg1: memref<256xf32>) {
  %cst = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
  %cst_0 = arith.constant dense<true> : vector<16xi1>
  %0 = xegpu.create_tdesc %arg1, %cst : memref<256xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>
  xegpu.store %arg0, %0, %cst_0  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.scatter_tdesc_attr<>>, vector<16xi1>
  return
}

// -----
// CHECK: function: test_vector_bitcast_i16_to_i8:
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xi16>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<32x16xi8>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xi32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xi16> -> !xegpu.tensor_desc<8x16xi16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<32x16xi8> -> !xegpu.tensor_desc<32x16xi8>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [4, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<8x16xi16> -> vector<8x16xi16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load_nd %[[T1]]  : !xegpu.tensor_desc<32x16xi8> -> vector<32x16xi8>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [4, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = vector.bitcast %[[T2]] : vector<8x16xi16> to vector<8x32xi8>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 2]
// CHECK-NEXT: op    : %[[T5:.*]] = xegpu.dpas %[[T4]], %[[T3]] : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T6:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_vector_bitcast_i16_to_i8(%arg0: memref<8x16xi16>, %arg1: memref<32x16xi8>, %arg2: memref<8x16xi32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xi16> -> !xegpu.tensor_desc<8x16xi16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<32x16xi8> -> !xegpu.tensor_desc<32x16xi8>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x16xi16> -> vector<8x16xi16>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<32x16xi8> -> vector<32x16xi8>
  %4 = vector.bitcast %2 : vector<8x16xi16> to vector<8x32xi8>
  %5 = xegpu.dpas %4, %3 : vector<8x32xi8>, vector<32x16xi8> -> vector<8x16xi32>
  %6 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xi32> -> !xegpu.tensor_desc<8x16xi32>
  xegpu.store_nd %5, %6  : vector<8x16xi32>, !xegpu.tensor_desc<8x16xi32>
  return
}

// -----
// CHECK: function: test_vector_bitcast_i8_to_f16:
// CHECK-NEXT: argument: <block argument> of type 'memref<8x32xi8>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<16x32xi8>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x32xi8> -> !xegpu.tensor_desc<8x32xi8>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 2]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<16x32xi8> -> !xegpu.tensor_desc<16x32xi8>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [4, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<8x32xi8> -> vector<8x32xi8>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 2]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load_nd %[[T1]]  : !xegpu.tensor_desc<16x32xi8> -> vector<16x32xi8>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [4, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = vector.bitcast %[[T2]] : vector<8x32xi8> to vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T5:.*]] = vector.bitcast %[[T3]] : vector<16x32xi8> to vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T6:.*]] = xegpu.dpas %[[T4]], %[[T5]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T7:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_vector_bitcast_i8_to_f16(%arg0: memref<8x32xi8>, %arg1: memref<16x32xi8>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x32xi8> -> !xegpu.tensor_desc<8x32xi8>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x32xi8> -> !xegpu.tensor_desc<16x32xi8>
  %2 = xegpu.load_nd %0  : !xegpu.tensor_desc<8x32xi8> -> vector<8x32xi8>
  %3 = xegpu.load_nd %1  : !xegpu.tensor_desc<16x32xi8> -> vector<16x32xi8>
  %4 = vector.bitcast %2 : vector<8x32xi8> to vector<8x16xf16>
  %5 = vector.bitcast %3 : vector<16x32xi8> to vector<16x16xf16>
  %6 = xegpu.dpas %4, %5 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %7 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %6, %7  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK: function: test_binary_op_one_use:
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = arith.addf %[[T1]], %[[T2]] : vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.dpas %[[T0]], %[[T3]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_binary_op_one_use(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: !xegpu.tensor_desc<8x16xf32>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %4, %arg2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK: function: test_binary_op_multiple_uses:
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16x16xf16>' at index: 3
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = arith.addf %[[T1]], %[[CST]] : vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.dpas %[[T0]], %[[T2]] : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_binary_op_multiple_uses(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: !xegpu.tensor_desc<8x16xf32>, %arg3: !xegpu.tensor_desc<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %cst = arith.constant dense<1.000000e+00> : vector<16x16xf16>
  %2 = arith.addf %1, %cst : vector<16x16xf16>
  %3 = xegpu.dpas %0, %2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %3, %arg2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %2, %arg3  : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  return
}

// -----
// CHECK: function: test_for_op:
// CHECK-NEXT: argument: <block argument> of type 'memref<8x128xf16>' at index: 0
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<128x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type 'memref<8x16xf32>' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 128 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %{{.*}} = arith.constant 16 : index
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x128xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T1:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<128x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T5:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T6:.*]] = xegpu.dpas %[[T4]], %[[T5]], %{{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T7:.*]] = xegpu.update_nd_offset %{{.*}} : !xegpu.tensor_desc<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T8:.*]] = xegpu.update_nd_offset %{{.*}} : !xegpu.tensor_desc<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : scf.for
// CHECK-NEXT: sg_map for result #0: Not assigned.
// CHECK-NEXT: sg_map for result #1: Not assigned.
// CHECK-NEXT: sg_map for result #2: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_for_op(%arg0: memref<8x128xf16>, %arg1: memref<128x16xf16>, %arg2: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x128xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<128x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  %2:3 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %0, %arg5 = %1, %arg6 = %cst) -> (!xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>) {
    %4 = xegpu.load_nd %arg4  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
    %5 = xegpu.load_nd %arg5  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %6 = xegpu.dpas %4, %5, %arg6 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
    %7 = xegpu.update_nd_offset %arg4, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16>
    %8 = xegpu.update_nd_offset %arg5, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16>
    scf.yield %7, %8, %6 : !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, vector<8x16xf32>
  }
  %3 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %2#2, %3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK: function: test_if_single_use:
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: argument: <block argument> of type 'i1' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf32>' at index: 3
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : scf.if
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [2, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.dpas %[[T0]], %{{.*}} : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_if_single_use(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1, %arg3: !xegpu.tensor_desc<8x16xf32>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %2, %arg3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  return
}

// -----
// CHECK: function: test_if_multiple_uses:
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf16>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16x16xf16>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type 'i1' at index: 2
// CHECK-NEXT: sg_map  : Not assigned.
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<8x16xf32>' at index: 3
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16x16xf16>' at index: 4
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T0:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T3:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T4:.*]] = xegpu.load_nd %{{.*}}  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : scf.if
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: op    : %[[T2:.*]] = xegpu.dpas %[[T0]], %{{.*}} : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [1, 16], wi_data: [1, 1]
func.func @test_if_multiple_uses(%arg0: !xegpu.tensor_desc<8x16xf16>, %arg1: !xegpu.tensor_desc<16x16xf16>, %arg2: i1, %arg3: !xegpu.tensor_desc<8x16xf32>, %arg4: !xegpu.tensor_desc<16x16xf16>) {
  %0 = xegpu.load_nd %arg0  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg2 -> (vector<16x16xf16>) {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  } else {
    %3 = xegpu.load_nd %arg1  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  xegpu.store_nd %2, %arg3  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %1, %arg4  : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  return
}

// -----
// CHECK: function: test_vector_outer_reduction:
// CHECK-NEXT: argument: <block argument> of type 'vector<16x16xf32>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16xf32>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[T0:.*]] = vector.multi_reduction <add>, %{{.*}}, %[[CST]] [0] : vector<16x16xf32> to vector<16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
func.func @test_vector_outer_reduction(%arg0: vector<16x16xf32>, %arg1: !xegpu.tensor_desc<16xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  %0 = vector.multi_reduction <add>, %arg0, %cst [0] : vector<16x16xf32> to vector<16xf32>
  xegpu.store_nd %0, %arg1  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  return
}

// -----
// CHECK: function: test_vector_inner_reduction:
// CHECK-NEXT: argument: <block argument> of type 'vector<16x16xf32>' at index: 0
// CHECK-NEXT: sg_map  : wi_layout: [1, 16], wi_data: [1, 1]
// CHECK-NEXT: argument: <block argument> of type '!xegpu.tensor_desc<16xf32>' at index: 1
// CHECK-NEXT: sg_map  : wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
// CHECK-NEXT: op    : %[[T0:.*]] = vector.multi_reduction <add>, %{{.*}}, %[[CST]] [1] : vector<16x16xf32> to vector<16xf32>
// CHECK-NEXT: sg_map for result #0: wi_layout: [16], wi_data: [1]
func.func @test_vector_inner_reduction(%arg0: vector<16x16xf32>, %arg1: !xegpu.tensor_desc<16xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  %0 = vector.multi_reduction <add>, %arg0, %cst [1] : vector<16x16xf32> to vector<16xf32>
  xegpu.store_nd %0, %arg1  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  return
}
