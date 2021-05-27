// RUN: mlir-opt -normalize-memrefs %s -split-input-file| FileCheck %s

// For all these cases, we test if MemRefs Normalization works with the test
// operations. These are test cases for MemRefs with dynamic dimension
// and tiled-layout map.
// * test.op_norm: this operation has the MemRefsNormalizable attribute. The tests
//   that include this operation are constructed so that the normalization should
//   happen.

#map_tiled = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 64, d2 mod 32, d3 mod 64)>

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2 ceildiv 32)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (32)>

// Test with op_norm and maps in arguments and in the operations in the function.
// Memref has two dynamic dimensions.

// CHECK-LABEL:  test_norm_dynamic12
// CHECK-SAME:   ([[ARG_0_:%.+]]: memref<1x?x?x1x?x64xf32>) {
func @test_norm_dynamic12(%arg0 : memref<1x?x?x14xf32, #map_tiled>) -> () {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.dim %arg0, %c1 :memref<1x?x?x14xf32, #map_tiled>
    %1 = memref.dim %arg0, %c2 :memref<1x?x?x14xf32, #map_tiled>
    %2 = memref.alloc(%0, %1) : memref<1x?x?x14xf32, #map_tiled>
    "test.op_norm"(%arg0, %2) : (memref<1x?x?x14xf32, #map_tiled>, memref<1x?x?x14xf32, #map_tiled>) -> ()
    memref.dealloc %2 :  memref<1x?x?x14xf32, #map_tiled>
    return
    // CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[ARG_0_]], [[CST_1_]] : memref<1x?x?x1x?x64xf32>
    // CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[ARG_0_]], [[CST_2_]] : memref<1x?x?x1x?x64xf32>
    // CHECK-DAG:       [[CST_1_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_14_:%.+]] = constant 14 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply #[[$MAP0]]([[CST_1_1_]], [[DIM_0_]], [[DIM_1_]], [[CST_14_]])
    // CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply #[[$MAP1]]([[CST_1_1_]], [[DIM_0_]], [[DIM_1_]], [[CST_14_]])
    // CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #[[$MAP2]]([[CST_1_1_]], [[DIM_0_]], [[DIM_1_]], [[CST_14_]])
    // CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_2_]], [[VAR_3_]], [[VAR_4_]]) : memref<1x?x?x1x?x64xf32>
    // CHECK:           "test.op_norm"([[ARG_0_]], [[RES_]]) : (memref<1x?x?x1x?x64xf32>, memref<1x?x?x1x?x64xf32>) -> ()
    // CHECK:           memref.dealloc [[RES_]] : memref<1x?x?x1x?x64xf32>
    // CHECK:           return
}

// -----

// Test with op_norm and maps in arguments and in the operations in the function.
// All of dimensions are dynamic.

#map_tiled1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, (d2 floordiv 4) floordiv 32, (d3 mod 8) floordiv 64, (d2 floordiv 4) mod 32, (d3 mod 8) mod 64)>

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> ((d2 floordiv 4) ceildiv 32)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (32)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> ((d3 mod 8) ceildiv 64)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (64)>

// CHECK-LABEL:  test_norm_dynamic1234
// CHECK-SAME:   ([[ARG_0_:%.+]]: memref<?x?x?x?x?x?xf32>) {
func @test_norm_dynamic1234(%arg0 : memref<?x?x?x?xf32, #map_tiled1>) -> () {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %0 = memref.dim %arg0, %c0 :memref<?x?x?x?xf32, #map_tiled1>
    %1 = memref.dim %arg0, %c1 :memref<?x?x?x?xf32, #map_tiled1>
    %2 = memref.dim %arg0, %c2 :memref<?x?x?x?xf32, #map_tiled1>
    %3 = memref.dim %arg0, %c3 :memref<?x?x?x?xf32, #map_tiled1>
    %4 = memref.alloc(%0, %1, %2, %3) : memref<?x?x?x?xf32, #map_tiled1>
    "test.op_norm"(%arg0, %4) : (memref<?x?x?x?xf32, #map_tiled1>, memref<?x?x?x?xf32, #map_tiled1>) -> ()
    memref.dealloc %4 :  memref<?x?x?x?xf32, #map_tiled1>
    return
    // CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
    // CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
    // CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[ARG_0_]], [[CST_0_]] : memref<?x?x?x?x?x?xf32>
    // CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[ARG_0_]], [[CST_1_]] : memref<?x?x?x?x?x?xf32>
    // CHECK-DAG:       [[DIM_2_:%.+]] = memref.dim [[ARG_0_]], [[CST_2_]] : memref<?x?x?x?x?x?xf32>
    // CHECK-DAG:       [[DIM_3_:%.+]] = memref.dim [[ARG_0_]], [[CST_3_]] : memref<?x?x?x?x?x?xf32>
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply #[[$MAP3]]([[DIM_0_]], [[DIM_1_]], [[DIM_2_]], [[DIM_3_]])
    // CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply #[[$MAP0]]([[DIM_0_]], [[DIM_1_]], [[DIM_2_]], [[DIM_3_]])
    // CHECK-DAG:       [[VAR_6_:%.+]] = affine.apply #[[$MAP1]]([[DIM_0_]], [[DIM_1_]], [[DIM_2_]], [[DIM_3_]])
    // CHECK-DAG:       [[VAR_7_:%.+]] = affine.apply #[[$MAP4]]([[DIM_0_]], [[DIM_1_]], [[DIM_2_]], [[DIM_3_]])
    // CHECK-DAG:       [[VAR_8_:%.+]] = affine.apply #[[$MAP2]]([[DIM_0_]], [[DIM_1_]], [[DIM_2_]], [[DIM_3_]])
    // CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply #[[$MAP5]]([[DIM_0_]], [[DIM_1_]], [[DIM_2_]], [[DIM_3_]])
    // CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_5_]], [[VAR_6_]], [[VAR_7_]], [[VAR_8_]], [[VAR_9_]]) : memref<?x?x?x?x?x?xf32>
    // CHECK:           "test.op_norm"([[ARG_0_]], [[RES_]]) : (memref<?x?x?x?x?x?xf32>, memref<?x?x?x?x?x?xf32>) -> ()
    // CHECK:           memref.dealloc [[RES_]] : memref<?x?x?x?x?x?xf32>
    // CHECK:           return
}

// -----

// Same test with maps that are not tiled layout maps in the arguments and the operations in the function.
// This is not normalized since this is not tiled-layout map. No mod and floordiv.

#map_not_tiled0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 - d2)>

// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 - d2)>

// CHECK-LABEL:  func @test_norm_dynamic_not_tiled0
// CHECK-SAME:   ([[ARG_0_:%.+]]: memref<1x?x?x14xf32, #[[$MAP6]]>) {
func @test_norm_dynamic_not_tiled0(%arg0 : memref<1x?x?x14xf32, #map_not_tiled0>) -> () {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.dim %arg0, %c1 :memref<1x?x?x14xf32, #map_not_tiled0>
    %1 = memref.dim %arg0, %c2 :memref<1x?x?x14xf32, #map_not_tiled0>
    %2 = memref.alloc(%0, %1) : memref<1x?x?x14xf32, #map_not_tiled0>
    "test.op_norm"(%arg0, %2) : (memref<1x?x?x14xf32, #map_not_tiled0>, memref<1x?x?x14xf32, #map_not_tiled0>) -> ()
    memref.dealloc %2 :  memref<1x?x?x14xf32, #map_not_tiled0>
    return
    // CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[ARG_0_]], [[CST_1_]] : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[ARG_0_]], [[CST_2_]] : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK:           [[RES_:%.+]] = memref.alloc([[DIM_0_]], [[DIM_1_]]) : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK:           "test.op_norm"([[ARG_0_]], [[RES_]]) : (memref<1x?x?x14xf32, #[[$MAP6]]>, memref<1x?x?x14xf32, #[[$MAP6]]>) -> ()
    // CHECK:           memref.dealloc [[RES_]] : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK:           return
}

// -----

// Same test with maps that are not tiled layout maps in the arguments and the operations in the function.
// This is not normalized since this is not tiled-layout map. No floordiv.

#map_not_tiled1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 - d2, d2 mod 32, d3 mod 64)>

// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 - d2, d2 mod 32, d3 mod 64)>

// CHECK-LABEL:  func @test_norm_dynamic_not_tiled1
// CHECK-SAME:   ([[ARG_0_:%.+]]: memref<1x?x?x14xf32, #[[$MAP6]]>) {
func @test_norm_dynamic_not_tiled1(%arg0 : memref<1x?x?x14xf32, #map_not_tiled1>) -> () {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.dim %arg0, %c1 :memref<1x?x?x14xf32, #map_not_tiled1>
    %1 = memref.dim %arg0, %c2 :memref<1x?x?x14xf32, #map_not_tiled1>
    %2 = memref.alloc(%0, %1) : memref<1x?x?x14xf32, #map_not_tiled1>
    "test.op_norm"(%arg0, %2) : (memref<1x?x?x14xf32, #map_not_tiled1>, memref<1x?x?x14xf32, #map_not_tiled1>) -> ()
    memref.dealloc %2 :  memref<1x?x?x14xf32, #map_not_tiled1>
    return
    // CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[ARG_0_]], [[CST_1_]] : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[ARG_0_]], [[CST_2_]] : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK:           [[RES_:%.+]] = memref.alloc([[DIM_0_]], [[DIM_1_]]) : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK:           "test.op_norm"([[ARG_0_]], [[RES_]]) : (memref<1x?x?x14xf32, #[[$MAP6]]>, memref<1x?x?x14xf32, #[[$MAP6]]>) -> ()
    // CHECK:           memref.dealloc [[RES_]] : memref<1x?x?x14xf32, #[[$MAP6]]>
    // CHECK:           return
}

// -----

// Same test with maps that are not tiled layout maps in the arguments and the operations in the function.
// This is not normalized since this is not tiled-layout map. RHS of floordiv is different from that of mod.

#map_not_tiled2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 floordiv 64, d2 mod 32, d3 mod 32)>

// CHECK-DAG: #[[$MAP7:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 floordiv 64, d2 mod 32, d3 mod 32)>

// CHECK-LABEL:  func @test_norm_dynamic_not_tiled2
// CHECK-SAME:   ([[ARG_0_:%.+]]: memref<1x?x?x14xf32, #[[$MAP7]]>) {
func @test_norm_dynamic_not_tiled2(%arg0 : memref<1x?x?x14xf32, #map_not_tiled2>) -> () {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.dim %arg0, %c1 :memref<1x?x?x14xf32, #map_not_tiled2>
    %1 = memref.dim %arg0, %c2 :memref<1x?x?x14xf32, #map_not_tiled2>
    %2 = memref.alloc(%0, %1) : memref<1x?x?x14xf32, #map_not_tiled2>
    "test.op_norm"(%arg0, %2) : (memref<1x?x?x14xf32, #map_not_tiled2>, memref<1x?x?x14xf32, #map_not_tiled2>) -> ()
    memref.dealloc %2 :  memref<1x?x?x14xf32, #map_not_tiled2>
    return
    // CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[ARG_0_]], [[CST_1_]] : memref<1x?x?x14xf32, #[[$MAP7]]>
    // CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[ARG_0_]], [[CST_2_]] : memref<1x?x?x14xf32, #[[$MAP7]]>
    // CHECK:           [[RES_:%.+]] = memref.alloc([[DIM_0_]], [[DIM_1_]]) : memref<1x?x?x14xf32, #[[$MAP7]]>
    // CHECK:           "test.op_norm"([[ARG_0_]], [[RES_]]) : (memref<1x?x?x14xf32, #[[$MAP7]]>, memref<1x?x?x14xf32, #[[$MAP7]]>) -> ()
    // CHECK:           memref.dealloc [[RES_]] : memref<1x?x?x14xf32, #[[$MAP7]]>
    // CHECK:           return
}

// -----

// Same test with maps that are not tiled layout maps in the arguments and the operations in the function.
// This is not normalized since this is not tiled-layout map. Multiple mod with the same LHS and RHS.

#map_not_tiled3 = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 32, d2, d3, d1 mod 32, d1 mod 32)>

// CHECK-DAG: #[[$MAP8:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 32, d2, d3, d1 mod 32, d1 mod 32)>

// CHECK-LABEL:  func @test_norm_dynamic_not_tiled3
// CHECK-SAME:   ([[ARG_0_:%.+]]: memref<1x?x?x14xf32, #[[$MAP8]]>) {
func @test_norm_dynamic_not_tiled3(%arg0 : memref<1x?x?x14xf32, #map_not_tiled3>) -> () {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.dim %arg0, %c1 :memref<1x?x?x14xf32, #map_not_tiled3>
    %1 = memref.dim %arg0, %c2 :memref<1x?x?x14xf32, #map_not_tiled3>
    %2 = memref.alloc(%0, %1) : memref<1x?x?x14xf32, #map_not_tiled3>
    "test.op_norm"(%arg0, %2) : (memref<1x?x?x14xf32, #map_not_tiled3>, memref<1x?x?x14xf32, #map_not_tiled3>) -> ()
    memref.dealloc %2 :  memref<1x?x?x14xf32, #map_not_tiled3>
    return
    // CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[ARG_0_]], [[CST_1_]] : memref<1x?x?x14xf32, #[[$MAP8]]>
    // CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[ARG_0_]], [[CST_2_]] : memref<1x?x?x14xf32, #[[$MAP8]]>
    // CHECK:           [[RES_:%.+]] = memref.alloc([[DIM_0_]], [[DIM_1_]]) : memref<1x?x?x14xf32, #[[$MAP8]]>
    // CHECK:           "test.op_norm"([[ARG_0_]], [[RES_]]) : (memref<1x?x?x14xf32, #[[$MAP8]]>, memref<1x?x?x14xf32, #[[$MAP8]]>) -> ()
    // CHECK:           memref.dealloc [[RES_]] : memref<1x?x?x14xf32, #[[$MAP8]]>
    // CHECK:           return
}

// -----

// Same test with maps that are not tiled layout maps in the arguments and the operations in the function.
// This is not normalized since this is not tiled-layout map. floordiv and mod with the same LHS and RHS(d0 floordiv 32 and d0 mod 32), but, unrelaed d0 exists in other position.

#map_not_tiled4 = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 32, d1 floordiv 32, d0, d3, d0 mod 32, d1 mod 32)>

// CHECK-DAG: #[[$MAP9:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 32, d1 floordiv 32, d0, d3, d0 mod 32, d1 mod 32)>

// CHECK-LABEL:  func @test_norm_dynamic_not_tiled4
// CHECK-SAME:   ([[ARG_0_:%.+]]: memref<1x?x?x14xf32, #[[$MAP9]]>) {
func @test_norm_dynamic_not_tiled4(%arg0 : memref<1x?x?x14xf32, #map_not_tiled4>) -> () {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.dim %arg0, %c1 :memref<1x?x?x14xf32, #map_not_tiled4>
    %1 = memref.dim %arg0, %c2 :memref<1x?x?x14xf32, #map_not_tiled4>
    %2 = memref.alloc(%0, %1) : memref<1x?x?x14xf32, #map_not_tiled4>
    "test.op_norm"(%arg0, %2) : (memref<1x?x?x14xf32, #map_not_tiled4>, memref<1x?x?x14xf32, #map_not_tiled4>) -> ()
    memref.dealloc %2 :  memref<1x?x?x14xf32, #map_not_tiled4>
    return
    // CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
    // CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
    // CHECK-NOT: separator of consecutive DAGs
    // CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[ARG_0_]], [[CST_1_]] : memref<1x?x?x14xf32, #[[$MAP9]]>
    // CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[ARG_0_]], [[CST_2_]] : memref<1x?x?x14xf32, #[[$MAP9]]>
    // CHECK:           [[RES_:%.+]] = memref.alloc([[DIM_0_]], [[DIM_1_]]) : memref<1x?x?x14xf32, #[[$MAP9]]>
    // CHECK:           "test.op_norm"([[ARG_0_]], [[RES_]]) : (memref<1x?x?x14xf32, #[[$MAP9]]>, memref<1x?x?x14xf32, #[[$MAP9]]>) -> ()
    // CHECK:           memref.dealloc [[RES_]] : memref<1x?x?x14xf32, #[[$MAP9]]>
    // CHECK:           return
}
