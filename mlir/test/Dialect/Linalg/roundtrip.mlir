// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// TODO: Re-enable LLVM lowering test.
//
// Test that we can lower all the way to LLVM without crashing, don't check results here.
// DISABLED: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[$id_2d:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$id_1d:.*]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-DAG: #[[$permute_0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[$permute_1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
// CHECK-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[$strided3DT:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 * s1 + s0 + d1 * s2 + d0)>

func @pad_dynamic(%arg0: tensor<1x2x2x?xf32>, %low: index, %high: index,
                  %pad_value: f32) -> tensor<6x?x?x?xf32> {
  %0 = linalg.pad_tensor %arg0 low[2, %low, 3, 3] high[3, 3, %high, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      linalg.yield %pad_value : f32
    } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
  return %0 : tensor<6x?x?x?xf32>
}
// CHECK-LABEL: func @pad_dynamic
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[LOW:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[HIGH:[a-zA-Z0-9_]*]]
//       CHECK:   linalg.pad_tensor %[[ARG0]]
//  CHECK-SAME:     low[2, %[[LOW]], 3, 3]
//  CHECK-SAME:     high[3, 3, %[[HIGH]], 2]
//       CHECK:    : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>

// -----

func @pad_static(%arg0: tensor<3x4xf32>, %pad_value: f32) -> tensor<6x9xf32> {
  %0 = linalg.pad_tensor %arg0 low[1, 2] high[2, 3] {
    ^bb0(%arg1 : index, %arg2 : index):
      linalg.yield %pad_value : f32
    } : tensor<3x4xf32> to tensor<6x9xf32>
  return %0 : tensor<6x9xf32>
}
// CHECK-LABEL: func @pad_static
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//       CHECK:   linalg.pad_tensor %[[ARG0]] low[1, 2] high[2, 3]
//       CHECK:    : tensor<3x4xf32> to tensor<6x9xf32>

// -----

func @pad_asymmetrical(%arg0: tensor<2x3xf32>, %ub0: index, %ub1: index,
                       %pad_value: f32) -> tensor<?x?xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[%ub0, %ub1] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad_value : f32
    } : tensor<2x3xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pad_asymmetrical
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB1:[a-zA-Z0-9_]*]]
//       CHECK:   linalg.pad_tensor %[[ARG0]]
//  CHECK-SAME:     low[0, 0]
//  CHECK-SAME:     high[%[[UB0]], %[[UB1]]]
//       CHECK:    : tensor<2x3xf32> to tensor<?x?xf32>

// -----

func @pad_to_static_size(%arg0: tensor<?x?xf32>, %ub0: index, %ub1: index,
                         %pad_value: f32) -> tensor<2x3xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[%ub0, %ub1] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-LABEL: func @pad_to_static_size
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB1:[a-zA-Z0-9_]*]]
//       CHECK:   linalg.pad_tensor %[[ARG0]]
//  CHECK-SAME:     low[0, 0]
//  CHECK-SAME:     high[%[[UB0]], %[[UB1]]]
//       CHECK:    : tensor<?x?xf32> to tensor<2x3xf32>

// -----

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


func @copy_view(%arg0: memref<?xf32, offset: ?, strides: [1]>,
                %arg1: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>,
                              memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @copy_view(
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) :
//  CHECK-SAME:     memref<?xf32, #[[$strided1D]]>, memref<?xf32, #[[$strided1D]]>

// -----


func @copy_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
                 %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = affine_map<(i, j, k) -> (i, k, j)>,
                             outputPermutation = affine_map<(i, j, k) -> (k, j, i)>} :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>) {
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) {
//  CHECK-SAME:     inputPermutation = #[[$permute_0]],
//  CHECK-SAME:     outputPermutation = #[[$permute_1]]} :
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]>,
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]>

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
   ^bb0(%arg3: f32):  // no predecessors
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
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32):  // no predecessors
    %5 = arith.cmpi sge, %arg3, %arg5 : i32
    %6 = select %5, %arg3, %arg5 : i32
    %7 = arith.cmpi eq, %arg3, %arg5 : i32
    %8 = arith.cmpi slt, %arg4, %arg6 : i32
    %9 = select %8, %arg4, %arg6 : i32
    %10 = select %5, %arg4, %arg6 : i32
    %11 = select %7, %9, %10 : i32
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

func @init_tensor(%arg0 : index, %arg1 : index)
{
  %0 = linalg.init_tensor [3, 42] : tensor<3x42xf32>
  %1 = linalg.init_tensor [4, %arg0, %arg1, 5] : tensor<4x?x?x5xf32>
  return
}
// CHECK-LABEL: func @init_tensor
//       CHECK:   linalg.init_tensor [3, 42] : tensor<3x42xf32>
//       CHECK:   linalg.init_tensor [4, %{{.*}}, %{{.*}}, 5] : tensor<4x?x?x5xf32>

// -----

func @fill_tensor(%arg0 : index, %arg1 : index, %arg2 : f32) -> tensor<?x?xf32> {
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = linalg.fill(%arg2, %0) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK: %{{.+}} = linalg.fill(%{{.+}}, %{{.+}}) : f32, tensor<?x?xf32> -> tensor<?x?xf32>

// -----

#accesses_4 = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>
]

#trait_4 = {
  indexing_maps = #accesses_4,
  iterator_types = ["parallel", "parallel"]
}

func @tiled_loop(%lhs: tensor<24x64xi8>, %rhs: tensor<24x64xi8>,
                 %out: tensor<24x64xi8>) -> tensor<24x64xi8> {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %c4 = arith.constant 4 : index
 %c24 = arith.constant 24 : index
 %c64 = arith.constant 64 : index
 %prod = linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
      ins(%lhs_ = %lhs: tensor<24x64xi8>, %rhs_ = %rhs: tensor<24x64xi8>)
      outs(%out_ = %out: tensor<24x64xi8>) {
    %lhs_sub = tensor.extract_slice %lhs_[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>
    %rhs_sub = tensor.extract_slice %rhs_[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>
    %out_sub = tensor.extract_slice %out_[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>

    %sum = linalg.generic #trait_4
        ins(%lhs_sub, %rhs_sub : tensor<?x?xi8>, tensor<?x?xi8>)
        outs(%out_sub : tensor<?x?xi8>) {
      ^bb(%l: i8, %r: i8, %o: i8) :
        %s = arith.addi %l, %r : i8
        linalg.yield %s : i8
      } -> tensor<?x?xi8>

    %sum_sub = tensor.insert_slice %sum into %out_[%i, 0][%c4, %c64][1, 1]
      : tensor<?x?xi8> into tensor<24x64xi8>
    linalg.yield %sum_sub : tensor<24x64xi8>
  }
  return %prod : tensor<24x64xi8>
}
// CHECK-LABEL: func @tiled_loop
// CHECK-NOT: iterators[

// -----

#id_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#id_2d = affine_map<(d0, d1, d2) -> (d0, d2)>
#id_1d = affine_map<(d0, d1, d2) -> (d1)>

#trait_5 = {
  indexing_maps = [
    #id_3d,
    #id_2d,
    #id_1d,
    #id_1d
  ],
  iterator_types = ["reduction", "parallel", "reduction"]
}

func @tiled_loop_reduction(%input_3d: tensor<16x24x32xf32>,
                           %input_2d: tensor<16x32xf32>,
                           %input_1d: tensor<24xf32>,
                           %output: tensor<24xf32>) -> tensor<24xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %X = tensor.dim %input_3d, %c0 : tensor<16x24x32xf32>
  %Y = tensor.dim %input_3d, %c1 : tensor<16x24x32xf32>
  %Z = tensor.dim %input_3d, %c2 : tensor<16x24x32xf32>
  %result = linalg.tiled_loop (%i, %j, %k)
      = (%c0, %c0, %c0) to (%X, %Y, %Z) step (%c2, %c4, %c8)
      ins(%i3d_ = %input_3d: tensor<16x24x32xf32>,
          %i2d_ = %input_2d: tensor<16x32xf32>,
          %i1d_ = %input_1d: tensor<24xf32>)
      outs(%o_ =  %output: tensor<24xf32>)
      iterators["reduction", "parallel", "reduction"]
      distribution["block_x", "block_y", "none"] {
    %sub_3d = tensor.extract_slice %i3d_[%i, %j, %k][2, 4, 8][1, 1, 1]
      : tensor<16x24x32xf32> to tensor<2x4x8xf32>
    %sub_2d = tensor.extract_slice %i2d_[%i, %k][2, 8][1, 1]
      : tensor<16x32xf32> to tensor<2x8xf32>
    %sub_1d = tensor.extract_slice %i1d_[%j] [4] [1]
      : tensor<24xf32> to tensor<4xf32>
    %sub_out = tensor.extract_slice %o_[%j] [4] [1]
      : tensor<24xf32> to tensor<4xf32>
    %acc = linalg.generic #trait_5
      ins(%sub_3d, %sub_2d, %sub_1d
        : tensor<2x4x8xf32>, tensor<2x8xf32>, tensor<4xf32>)
      outs(%sub_out : tensor<4xf32>)  {
    ^bb0(%i3d: f32, %i2d: f32, %i1d: f32, %o: f32):
      %0 = arith.addf %i3d, %i2d : f32
      %1 = arith.addf %0, %i1d : f32
      linalg.yield %1 : f32
    } -> tensor<4xf32>

    %sum_sub = tensor.insert_slice %acc into %o_[%j][4][1]
      : tensor<4xf32> into tensor<24xf32>
    linalg.yield %sum_sub : tensor<24xf32>
  }
  return %result : tensor<24xf32>
}
// CHECK-LABEL: func @tiled_loop_reduction
// CHECK: iterators[

// -----

#trait_6 = {
  indexing_maps = [
    #id_3d,
    #id_2d,
    #id_1d,
    #id_1d
  ],
  iterator_types = ["reduction", "parallel", "reduction"]
}
#map_1 = affine_map<(d0, d1, d2)[s0] -> (d0 * 768 + s0 + d1 * 32 + d2)>
#map_2 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>
#map_3 = affine_map<(d0)[s0] -> (d0 + s0)>

func @tiled_loop_on_buffers(%input_3d: memref<16x24x32xf32>,
                            %input_2d: memref<16x32xf32>,
                            %input_1d: memref<24xf32>,
                            %output: memref<24xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %X = memref.dim %input_3d, %c0 : memref<16x24x32xf32>
  %Y = memref.dim %input_3d, %c1 : memref<16x24x32xf32>
  %Z = memref.dim %input_3d, %c2 : memref<16x24x32xf32>
  linalg.tiled_loop (%i, %j, %k) = (%c0, %c0, %c0)
      to (%X, %Y, %Z) step (%c2, %c4, %c8)
      ins(%i3d_ = %input_3d: memref<16x24x32xf32>,
          %i2d_ = %input_2d: memref<16x32xf32>,
          %i1d_ = %input_1d: memref<24xf32>)
      outs(%o_ =  %output: memref<24xf32>)
      iterators["reduction", "parallel", "reduction"] {
    %sub_3d = memref.subview %i3d_[%i, %j, %k][2, 4, 8][1, 1, 1]
      : memref<16x24x32xf32> to memref<2x4x8xf32, #map_1>
    %sub_2d = memref.subview %i2d_[%i, %k][2, 8][1, 1]
      : memref<16x32xf32> to memref<2x8xf32, #map_2>
    %sub_1d = memref.subview %i1d_[%j] [4] [1]
      : memref<24xf32> to memref<4xf32, #map_3>
    %sub_out = memref.subview %o_[%j] [4] [1]
      : memref<24xf32> to memref<4xf32, #map_3>
    linalg.generic #trait_6
      ins(%sub_3d, %sub_2d, %sub_1d
        : memref<2x4x8xf32, #map_1>,
          memref<2x8xf32, #map_2>,
          memref<4xf32, #map_3>)
      outs(%sub_out : memref<4xf32, #map_3>)  {
    ^bb0(%i3d: f32, %i2d: f32, %i1d: f32, %o: f32):
      %0 = arith.addf %i3d, %i2d : f32
      %1 = arith.addf %0, %i1d : f32
      linalg.yield %1 : f32
    }
    linalg.yield
  }
  return
}
// CHECK-LABEL: func @tiled_loop_on_buffers
// CHECK: iterators[
