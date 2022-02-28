// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// TODO: Re-enable LLVM lowering test.
//
// Test that we can lower all the way to LLVM without crashing, don't check results here.
// DISABLED: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[$strided3DT:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 * s1 + s0 + d1 * s2 + d0)>

func @views(%arg0: index) {
  %c0 = arith.constant 0 : index
  %0 = arith.muli %arg0, %arg0 : index
  %1 = memref.alloc (%0) : memref<?xi8>
  %3 = memref.view %1[%c0][%arg0, %arg0] : memref<?xi8> to memref<?x?xf32>
  %4 = memref.view %1[%c0][%arg0, %arg0] : memref<?xi8> to memref<?x?xvector<4x4xf32>>
  memref.dealloc %1 : memref<?xi8>
  return
}
// CHECK-LABEL: func @views
//  CHECK:  arith.muli %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:  memref.alloc(%{{.*}}) : memref<?xi8>
//  CHECK-NEXT:  memref.view %{{.*}}[%{{.*}}][%{{.*}}] :
//  CHECK-SAME:     memref<?xi8> to memref<?x?xf32>
//  CHECK-NEXT:  memref.view %{{.*}}[%{{.*}}][%{{.*}}] :
//  CHECK-SAME:     memref<?xi8> to memref<?x?xvector<4x4xf32>>
//  CHECK-NEXT:  memref.dealloc %{{.*}} : memref<?xi8>

// -----

func @ops(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>,
          %arg1: memref<?xf32, offset: ?, strides: [1]>,
          %arg2: memref<?xf32, offset: ?, strides: [1]>,
          %arg3: memref<f32>) {
  linalg.matmul ins(%arg0, %arg0 : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                                   memref<?x?xf32, offset: ?, strides: [?, 1]>)
               outs(%arg0 : memref<?x?xf32, offset: ?, strides: [?, 1]>)
  linalg.matvec ins(%arg0, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                                  memref<?xf32, offset: ?, strides: [1]>)
               outs(%arg2: memref<?xf32, offset: ?, strides: [1]>)
  linalg.dot ins(%arg1, %arg2: memref<?xf32, offset: ?, strides: [1]>,
                               memref<?xf32, offset: ?, strides: [1]>)
            outs(%arg3: memref<f32>)
  return
}
// CHECK-LABEL: func @ops(%
// CHECK: linalg.matmul
// CHECK-SAME:   ins(%{{.*}}, %{{.*}} : memref<?x?xf32, #[[$strided2D]]>,
// CHECK-SAME:                          memref<?x?xf32, #[[$strided2D]]>)
// CHECK-SAME:  outs(%{{.*}} : memref<?x?xf32, #[[$strided2D]]>)
// CHECK: linalg.matvec
// CHECK-SAME:   ins(%{{.*}}, %{{.*}}: memref<?x?xf32, #[[$strided2D]]>,
// CHECK-SAME:                         memref<?xf32, #[[$strided1D]]>)
// CHECK-SAME:  outs(%{{.*}}: memref<?xf32, #[[$strided1D]]>)
// CHECK: linalg.dot
// CHECK-SAME:   ins(%{{.*}}, %{{.*}}: memref<?xf32, #[[$strided1D]]>,
// CHECK-SAME:                         memref<?xf32, #[[$strided1D]]>)
// CHECK-SAME:  outs(%{{.*}}: memref<f32>)

// -----

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg1, %arg0) : f32, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK:  %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : f32, memref<?xf32, #[[$strided1D]]>

// -----

func @transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %0 = memref.transpose %arg0 (i, j, k) -> (k, j, i) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 * s1 + s0 + d1 * s2 + d0)>>
  return
}
// CHECK-LABEL: func @transpose
//       CHECK:   memref.transpose %{{.*}} ([[i:.*]], [[j:.*]], [[k:.*]]) -> ([[k]], [[j]], [[i]]) :
//  CHECK-SAME:      memref<?x?x?xf32, #[[$strided3D]]> to memref<?x?x?xf32, #[[$strided3DT]]>

// -----


func @fill_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: f32) {
  linalg.fill(%arg1, %arg0) : f32, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : f32, memref<?x?x?xf32, #[[$strided3D]]>

// -----

#accesses_0 = [
  affine_map<(i, j, k) -> (j, i)>,
  affine_map<(i, j, k) -> ()>,
  affine_map<(i, j, k) -> (i, k, i + j)>
]

#trait_0 = {
  indexing_maps = #accesses_0,
  iterator_types = ["parallel", "parallel", "parallel"],
  library_call = "some_external_function_name_1"
}

func @generic(%arg0: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>,
              %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %cst = arith.constant 0.0 : f32
  linalg.generic #trait_0
       ins(%arg0, %cst : memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>, f32)
      outs(%arg1 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>)
      attrs = {foo = 1} {
    ^bb(%0: vector<3x4xi4>, %1: f32, %2: f32) :
      linalg.yield %1 : f32
  }
  return
}
// CHECK-LABEL: func @generic
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [#{{[0-9a-z]*}}, #{{[0-9a-z]*}}, #{{[0-9a-z]*}}],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_1"}
//  CHECK-SAME:      ins({{.*}}, {{.*}} : memref<?x?xvector<3x4xi4>, #[[$strided2D]]>, f32)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, #[[$strided3D]]>)
//  CHECK-SAME:     {foo = 1 : i64}

func @generic_with_tensor_input(%arg0: tensor<?x?xvector<3x4xi4>>,
                                %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %cst = arith.constant 0.0 : f32
  linalg.generic #trait_0
       ins(%arg0, %cst : tensor<?x?xvector<3x4xi4>>, f32)
      outs(%arg1 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>)
      attrs = {foo = 1} {
    ^bb(%0: vector<3x4xi4>, %1: f32, %2: f32) :
      linalg.yield %1 : f32
  }
  return
}
// CHECK-LABEL: func @generic_with_tensor_input
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_1"}
//  CHECK-SAME:     ins({{.*}}, {{.*}} : tensor<?x?xvector<3x4xi4>>, f32)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, #[[$strided3D]]>)
//  CHECK-SAME:     {foo = 1 : i64}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_without_inputs(%arg0 : memref<?x?x?xf32>) {
  linalg.generic  {indexing_maps = [#map0],
                   iterator_types = ["parallel", "parallel", "parallel"]}
                  outs(%arg0 : memref<?x?x?xf32>) {
   ^bb0(%arg3: f32):
      %cst = arith.constant 0.000000e+00 : f32
      linalg.yield %cst : f32
    }
  return
}

// CHECK-LABEL: func @generic_without_inputs
//       CHECK:   linalg.generic
//   CHECK-NOT:     ins

// -----

#accesses_1 = [
  affine_map<(i, j, k) -> (j, i)>,
  affine_map<(i, j, k) -> (i, k, i + j)>,
  affine_map<(i, j, k) -> (i, k, i + j)>
]

#trait_1 = {
  indexing_maps = #accesses_1,
  iterator_types = ["parallel", "parallel", "parallel"],
  library_call = "some_external_function_name_1"
}

func @generic_with_tensor_input_and_output(
    %arg0: tensor<?x?xvector<3x4xi4>>, %arg1: tensor<?x?x?xf32>)
    -> (tensor<?x?x?xf32>) {
  %0 = linalg.generic #trait_1
       ins(%arg0, %arg1 : tensor<?x?xvector<3x4xi4>>, tensor<?x?x?xf32>)
      outs(%arg1 : tensor<?x?x?xf32>)
      attrs = {foo = 1} {
    ^bb(%0: vector<3x4xi4>, %1: f32, %2: f32) :
      %f0 = arith.constant 0.0 : f32
      linalg.yield %f0 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @generic_with_tensor_input_and_output
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_1"}
//  CHECK-SAME:      ins({{.*}} : tensor<?x?xvector<3x4xi4>>, tensor<?x?x?xf32>)
//  CHECK-SAME:     outs({{.*}} : tensor<?x?x?xf32>)
//  CHECK-SAME:     {foo = 1 : i64}
//       CHECK:     -> tensor<?x?x?xf32>
//       CHECK:   return {{.*}} : tensor<?x?x?xf32>

// -----

func @generic_with_multiple_tensor_outputs(
    %arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: i32)
    -> (tensor<i32>, tensor<i32>) {
  %c0 = arith.constant 0 : index
  %0 = linalg.init_tensor [] : tensor<i32>
  %1 = linalg.fill(%arg2, %0) : i32, tensor<i32> -> tensor<i32>
  %2 = linalg.init_tensor [] : tensor<i32>
  %3 = linalg.fill(%arg2, %2) : i32, tensor<i32> -> tensor<i32>
  %4:2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%arg0, %arg1 : tensor<?xi32>, tensor<?xi32>)
    outs(%1, %3 : tensor<i32>, tensor<i32>) {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32):
    %5 = arith.cmpi sge, %arg3, %arg5 : i32
    %6 = arith.select %5, %arg3, %arg5 : i32
    %7 = arith.cmpi eq, %arg3, %arg5 : i32
    %8 = arith.cmpi slt, %arg4, %arg6 : i32
    %9 = arith.select %8, %arg4, %arg6 : i32
    %10 = arith.select %5, %arg4, %arg6 : i32
    %11 = arith.select %7, %9, %10 : i32
    linalg.yield %6, %11 : i32, i32
  } -> (tensor<i32>, tensor<i32>)
  return %4#0, %4#1 : tensor<i32>, tensor<i32>
}
// CHECK-LABEL: func @generic_with_multiple_tensor_outputs
//       CHECK:   %{{.*}} = linalg.generic {
//  CHECK-SAME:      ins({{.*}} : tensor<?xi32>, tensor<?xi32>)
//  CHECK-SAME:     outs({{.*}} : tensor<i32>, tensor<i32>)
//       CHECK:   } -> (tensor<i32>, tensor<i32>)

// -----

#broadcast_access = [
  affine_map<(i, j) -> ()>,
  affine_map<(i, j) -> (i, j)>
]

#trait_broadcast = {
  indexing_maps = #broadcast_access,
  iterator_types = ["parallel", "parallel"],
  library_call = "some_broadcast_external_fn"
}

func @generic_op_zero_rank(%arg0: tensor<f32>, %arg1 : tensor<3x4xf32>) -> (tensor<3x4xf32>)
{
  %0 = linalg.generic #trait_broadcast
       ins(%arg0 : tensor<f32>)
      outs(%arg1 : tensor<3x4xf32>) {
    ^bb(%a: f32, %b: f32) :
      linalg.yield %a : f32
  } -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----


#accesses_3 = [
  affine_map<(i, j, k) -> (j, i)>,
  affine_map<(i, j, k) -> (i, k, i + j)>
]

#trait_3 = {
  indexing_maps = #accesses_3,
  iterator_types = ["parallel", "parallel", "parallel"],
  library_call = "some_external_function_name_2"
}

func @generic_region(%arg0: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>,
                     %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait_3
       ins(%arg0 : memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>)
      outs(%arg1 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>)
      attrs = {foo = 1} {
    ^bb(%a: vector<3x4xi4>, %b: f32) :
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      linalg.yield %b : f32
  }
  return
}
// CHECK-LABEL: func @generic_region
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [#{{[0-9a-z]*}}, #{{[0-9a-z]*}}],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_2"
//  CHECK-SAME:      ins({{.*}} : memref<?x?xvector<3x4xi4>, #[[$strided2D]]>)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, #[[$strided3D]]>)
//  CHECK-SAME:     attrs = {foo = 1 : i64} {
//       CHECK:  ^{{.*}}(%{{.*}}: vector<3x4xi4>, %{{.*}}: f32):
//       CHECK:    %{{.*}} = linalg.index 0 : index
//       CHECK:    %{{.*}} = linalg.index 1 : index
//       CHECK:    %{{.*}} = linalg.index 2 : index
//       CHECK:    linalg.yield %{{.*}} : f32

// -----


func @named_ops(%a3: memref<?x?x?xf32>, %b3: memref<?x?x?xf32>, %c3: memref<?x?x?xf32>,
                %ta3: tensor<?x?x?xf32>, %tb3: tensor<?x?x?xf32>, %tc3: tensor<?x?x?xf32>)
  -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
{
  linalg.batch_matmul ins(%a3, %b3: memref<?x?x?xf32>, memref<?x?x?xf32>)
                     outs(%c3: memref<?x?x?xf32>)
  linalg.batch_matmul ins(%ta3, %tb3: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
                     outs(%c3: memref<?x?x?xf32>)
  %res1 = linalg.batch_matmul
                      ins(%ta3, %tb3: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
                     outs(%tc3: tensor<?x?x?xf32>)
                  -> tensor<?x?x?xf32>
  %res2 = linalg.batch_matmul
                      ins(%ta3, %b3: tensor<?x?x?xf32>, memref<?x?x?xf32>)
                     outs(%tc3: tensor<?x?x?xf32>)
                  -> tensor<?x?x?xf32>
  return %res1, %res2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}
// CHECK-LABEL: func @named_ops
//       CHECK:   linalg.batch_matmul
//       CHECK:   linalg.batch_matmul
//       CHECK:   linalg.batch_matmul
//       CHECK:   linalg.batch_matmul

// -----

#attr = {"foo"}
func @init_tensor(%arg0 : index, %arg1 : index)
{
  %0 = linalg.init_tensor [3, 42] : tensor<3x42xf32>
  %1 = linalg.init_tensor [4, %arg0, %arg1, 5] : tensor<4x?x?x5xf32>
  %2 = linalg.init_tensor [2, 2] : tensor<2x2xf32, #attr>
  return
}
// CHECK-LABEL: func @init_tensor
//       CHECK:   linalg.init_tensor [3, 42] : tensor<3x42xf32>
//       CHECK:   linalg.init_tensor [4, %{{.*}}, %{{.*}}, 5] : tensor<4x?x?x5xf32>
//       CHECK:   linalg.init_tensor [2, 2] : tensor<2x2xf32, {foo}>

// -----

func @fill_tensor(%arg0 : index, %arg1 : index, %arg2 : f32) -> tensor<?x?xf32> {
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = linalg.fill(%arg2, %0) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK: %{{.+}} = linalg.fill(%{{.+}}, %{{.+}}) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
