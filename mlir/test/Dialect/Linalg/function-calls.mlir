// RUN: mlir-opt %s -convert-linalg-to-function-calls -split-input-file --verify-diagnostics | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

// CHECK:  func private @linalg_fill_f32_viewsxsxf32(f32, memref<?x?xf32, {{.*}}>) attributes {llvm.emit_c_interface}
// CHECK:  func private @linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32(memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>) attributes {llvm.emit_c_interface}

func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>) -> (memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %x = memref.dim %A, %c0 : memref<?x?xf32>
  %y = memref.dim %B, %c1 : memref<?x?xf32>
  %C = memref.alloc(%x, %y) : memref<?x?xf32>

  // CHECK: call @linalg_fill_f32_viewsxsxf32({{.*}}) : (f32, memref<?x?xf32, {{.*}}>)
  linalg.fill ins(%f0 : f32) outs(%C : memref<?x?xf32>)

  // CHECK:  call @linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32({{.*}}) : (memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>, memref<?x?xf32, {{.*}}>) -> ()
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
                outs(%C: memref<?x?xf32>)
  return %C : memref<?x?xf32>
}

// -----

#accesses = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
    ]
#trait = {
    doc = "...",
    indexing_maps = #accesses,
    library_call = "test",
    iterator_types = ["parallel"]
}

// CHECK: func.func private @linalg_copy_view32xf16as1_view32xf16as6(memref<32xf16, strided<[?], offset: ?>, 1>, memref<32xf16, strided<[?], offset: ?>, 6>) attributes {llvm.emit_c_interface}
// CHECK: func.func private @linalg_copy_view32xf16as6_view32xf16as1(memref<32xf16, strided<[?], offset: ?>, 6>, memref<32xf16, strided<[?], offset: ?>, 1>) attributes {llvm.emit_c_interface}

module {
  func.func @helper(%arg7: memref<32xf16, 1>, %arg8: memref<32xf16, 1>, %arg9: memref<32xf16, 1>) {
    %localA = memref.alloca() : memref<32xf16, 6>
    %localB = memref.alloca() : memref<32xf16, 6>
    %localOut = memref.alloca() : memref<32xf16, 6>
    linalg.copy ins(%arg8 : memref<32xf16, 1>) outs(%localA : memref<32xf16, 6>)
    linalg.copy ins(%arg9 : memref<32xf16, 1>) outs(%localB : memref<32xf16, 6>)

    linalg.generic #trait
    ins(%localA, %localB : memref<32xf16, 6>, memref<32xf16, 6>)
    outs(%localOut : memref<32xf16, 6>) {
      ^bb0(%0: f16, %1: f16, %2: f16) :
        %e = arith.addf %1, %0: f16
        linalg.yield %e : f16
    }

    linalg.copy ins(%localOut : memref<32xf16, 6>) outs(%arg7 : memref<32xf16, 1>)
    return
  }
}


// -----

// CHECK: func.func private @linalg_elemwise_unary_negf_view16x8xf32_view16x8xf32(memref<16x8xf32, strided<[?, ?], offset: ?>>, memref<16x8xf32, strided<[?, ?], offset: ?>>) attributes {llvm.emit_c_interface}
// CHECK: func.func private @linalg_elemwise_unary_negf_view16xf32_view16xf32(memref<16xf32, strided<[?], offset: ?>>, memref<16xf32, strided<[?], offset: ?>>) attributes {llvm.emit_c_interface}

func.func @test_neg(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16xf32>, %D: memref<16xf32>) {
  linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
                              ins(%A: memref<16x8xf32>) outs(%B: memref<16x8xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
                              ins(%C: memref<16xf32>) outs(%D: memref<16xf32>)
  return
}

// -----

// CHECK: func.func private @linalg_elemwise_unary_exp_view16x8xf32_view16x8xf32(memref<16x8xf32, strided<[?, ?], offset: ?>>, memref<16x8xf32, strided<[?, ?], offset: ?>>) attributes {llvm.emit_c_interface}
// CHECK: func.func private @linalg_elemwise_unary_exp_view16xf32_view16xf32(memref<16xf32, strided<[?], offset: ?>>, memref<16xf32, strided<[?], offset: ?>>) attributes {llvm.emit_c_interface}

func.func @test_exp(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16xf32>, %D: memref<16xf32>) {
  linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
                              ins(%A: memref<16x8xf32>) outs(%B: memref<16x8xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
                              ins(%C: memref<16xf32>) outs(%D: memref<16xf32>)
  return
}

// -----

// CHECK: func.func private @linalg_elemwise_binary_add_view16x8xf32_view16x8xf32_view16x8xf32(memref<16x8xf32, strided<[?, ?], offset: ?>>, memref<16x8xf32, strided<[?, ?], offset: ?>>, memref<16x8xf32, strided<[?, ?], offset: ?>>) attributes {llvm.emit_c_interface}
// CHECK: func.func private @linalg_elemwise_binary_add_view16xf32_view16xf32_view16xf32(memref<16xf32, strided<[?], offset: ?>>, memref<16xf32, strided<[?], offset: ?>>, memref<16xf32, strided<[?], offset: ?>>) attributes {llvm.emit_c_interface}

func.func @test_add(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>, %D: memref<16xf32>, %E: memref<16xf32>, %F: memref<16xf32>) {
  linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
                              ins(%A, %B: memref<16x8xf32>, memref<16x8xf32>) outs(%C: memref<16x8xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
                              ins(%D, %E: memref<16xf32>, memref<16xf32>) outs(%F: memref<16xf32>)
  return
}

// -----

func.func @dot(%arg0: memref<?xf32, strided<[1], offset: ?>>,
          %arg1: memref<?xf32, strided<[1], offset: ?>>,
          %arg2: memref<f32>) {
  linalg.dot ins(%arg0, %arg1: memref<?xf32, strided<[1], offset: ?>>,
                               memref<?xf32, strided<[1], offset: ?>>)
             outs(%arg2: memref<f32>)
  return
}
// CHECK-LABEL: func @dot(
//  CHECK-SAME: %[[arg0:[a-zA-z0-9]*]]: memref<?xf32, strided<[1], offset: ?>>,
//  CHECK-SAME: %[[arg1:[a-zA-z0-9]*]]: memref<?xf32, strided<[1], offset: ?>>,
//  CHECK-SAME: %[[arg2:[a-zA-z0-9]*]]: memref<f32>) {
//       CHECK:   %[[o0:.*]] = memref.cast %[[arg0]] :
//  CHECK-SAME:     memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   %[[o1:.*]] = memref.cast %[[arg1]] :
//  CHECK-SAME:     memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   %[[o2:.*]] = memref.cast %[[arg2]] :
//  CHECK-SAME:     memref<f32> to memref<f32, strided<[], offset: ?>>
//       CHECK:   call @linalg_dot_viewsxf32_viewsxf32_viewf32(
//  CHECK-SAME:     %[[o0]], %[[o1]], %[[o2]]) :
//  CHECK-SAME:   memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>, memref<f32, strided<[], offset: ?>>

// -----

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses,
  library_call = "external_outerproduct_matmul"
}

!vector_type_A = vector<4xf32>
!vector_type_B = vector<4xf32>
!vector_type_C = vector<4x4xf32>

!matrix_type_A = memref<?x?x!vector_type_A>
!matrix_type_B = memref<?x?x!vector_type_B>
!matrix_type_C = memref<?x?x!vector_type_C>

func.func @matmul_vec_impl(%A: !matrix_type_A, %B: !matrix_type_B, %C: !matrix_type_C) {
  linalg.generic #matmul_trait
      ins(%A, %B : !matrix_type_A, !matrix_type_B)
     outs(%C : !matrix_type_C) {
    ^bb0(%a: !vector_type_A, %b: !vector_type_B, %c: !vector_type_C):
      %d = vector.outerproduct %a, %b, %c: !vector_type_A, !vector_type_B
      linalg.yield %d: !vector_type_C
  }
  return
}
// CHECK-LABEL: func @matmul_vec_impl(
// CHECK:  call @external_outerproduct_matmul(%{{.*}}) :

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @func(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>)  {
  // expected-error @below {{failed to legalize}}
  %0 = linalg.generic {
    indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
  ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32): 
    linalg.yield %in : f32
  } -> tensor<?xf32>
  return 
}

// -----

func.func @func(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @below {{failed to legalize}}
  %0 = linalg.copy ins(%arg0 : tensor<4x8xf32>) outs(%arg1 : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
