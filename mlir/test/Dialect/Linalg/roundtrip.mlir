// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// TODO: Re-enable LLVM lowering test after IndexedGenericOp is lowered.
//
// Test that we can lower all the way to LLVM without crashing, don't check results here.
// DISABLED: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[$permute_0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[$permute_1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
// CHECK-DAG: #[[$reshape5D01:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
// CHECK-DAG: #[[$reshape5D0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
// CHECK-DAG: #[[$reshape5D1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1)>
// CHECK-DAG: #[[$reshape5D2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d2)>
// CHECK-DAG: #[[$reshape5D345:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
// CHECK-DAG: #[[$reshape5D34:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-DAG: #[[$reshapeD012:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$reshapeD01:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[$reshapeD0:.*]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG: #[[$reshapeD12:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$reshapeD2:.*]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: #[[$strided1D:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$strided2DOFF0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
// CHECK-DAG: #[[$strided3D:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[$strided3DOFF0:.*]] = affine_map<(d0, d1, d2)[s0, s1] -> (d0 * s0 + d1 * s1 + d2)>
// CHECK-DAG: #[[$strided3DT:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 * s1 + s0 + d1 * s2 + d0)>
// CHECK-DAG: #[[$strided6D:.*]] = affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4 + d4 * s5 + d5)>

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

func @range(%arg0: index, %arg1: index, %arg2: index) {
  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  linalg.range %{{.*}} : %{{.*}} : %{{.*}} : !linalg.range

// -----

func @views(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %c0 = constant 0 : index
  %0 = muli %arg0, %arg0 : index
  %1 = memref.alloc (%0) : memref<?xi8>
  %2 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  %3 = memref.view %1[%c0][%arg0, %arg0] : memref<?xi8> to memref<?x?xf32>
  %4 = memref.view %1[%c0][%arg0, %arg0] : memref<?xi8> to memref<?x?xvector<4x4xf32>>
  memref.dealloc %1 : memref<?xi8>
  return
}
// CHECK-LABEL: func @views
//  CHECK:  muli %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:  memref.alloc(%{{.*}}) : memref<?xi8>
//  CHECK-NEXT:  range
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
  linalg.fill(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, f32
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK:  %{{.*}}: memref<?xf32, #[[$strided1D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : memref<?xf32, #[[$strided1D]]>, f32

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
  linalg.fill(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, f32
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[$strided3D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : memref<?x?x?xf32, #[[$strided3D]]>, f32

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


func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
                 %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
                 %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
                                     memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>,
                                     memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view3(
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) :
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]>,
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]>,
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]>

// -----


func @conv_view6(%arg0: memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>,
                 %arg1: memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>,
                 %arg2: memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} :
    memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>,
    memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>,
    memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view6(
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) {
//  CHECK-SAME:     dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} :
//  CHECK-SAME:     memref<?x?x?x?x?x?xf32, #[[$strided6D]]>,
//  CHECK-SAME:     memref<?x?x?x?x?x?xf32, #[[$strided6D]]>,
//  CHECK-SAME:     memref<?x?x?x?x?x?xf32, #[[$strided6D]]>

// -----

func @conv_padding(%arg0: memref<?x?x?x?xf32>,
                   %arg1: memref<?x?x?x?xf32>,
                   %arg2: memref<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1],
                                    padding = dense<[[0, 1], [1, 1]]> : tensor<2x2xi64>,
                                    strides = [1, 1]} :
    memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
  return
}

// CHECK-LABEL: func @conv_padding(
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) {
//  CHECK-SAME:     dilations = [1, 1],
//  CHECK-SAME:     padding = dense<[
//  CHECK-SAME:                      [0, 1], [1, 1]]> : tensor<2x2xi64>,
//  CHECK-SAME:     strides = [1, 1]} :
//  CHECK-SAME:     memref<?x?x?x?xf32>,
//  CHECK-SAME:     memref<?x?x?x?xf32>,
//  CHECK-SAME:     memref<?x?x?x?xf32>

// -----

func @pooling_max(%arg0: memref<?x?x?xf32>,
                  %arg1: memref<?x?x?xi32>,
                  %arg2: memref<?x?x?xf32>) {
  linalg.pooling_max(%arg0, %arg1, %arg2) {strides = [2, 1, 2]}:
    memref<?x?x?xf32>, memref<?x?x?xi32>, memref<?x?x?xf32>
  return
}
// CHECK-LABEL: func @pooling_max
//       CHECK:   linalg.pooling_max(%{{.*}}, %{{.*}}, %{{.*}})
//  CHECK-SAME:   {strides = [2, 1, 2]}
//  CHECK-SAME:   memref<?x?x?xf32>, memref<?x?x?xi32>, memref<?x?x?xf32>

// -----

func @pooling_min(%arg0: memref<?x?x?xf32>,
                  %arg1: memref<?x?x?xi32>,
                  %arg2: memref<?x?x?xf32>) {
  linalg.pooling_min(%arg0, %arg1, %arg2) {strides = [2, 1, 2]}:
    memref<?x?x?xf32>, memref<?x?x?xi32>, memref<?x?x?xf32>
  return
}
// CHECK-LABEL: func @pooling_min
//       CHECK:   linalg.pooling_min(%{{.*}}, %{{.*}}, %{{.*}})
//  CHECK-SAME:   {strides = [2, 1, 2]}
//  CHECK-SAME:   memref<?x?x?xf32>, memref<?x?x?xi32>, memref<?x?x?xf32>

// -----

func @pooling_sum(%arg0: memref<?x?x?xf32>,
                  %arg1: memref<?x?x?xi32>,
                  %arg2: memref<?x?x?xf32>) {
  linalg.pooling_sum(%arg0, %arg1, %arg2) {strides = [2, 1, 2]}:
    memref<?x?x?xf32>, memref<?x?x?xi32>, memref<?x?x?xf32>
  return
}
// CHECK-LABEL: func @pooling_sum
//       CHECK:   linalg.pooling_sum(%{{.*}}, %{{.*}}, %{{.*}})
//  CHECK-SAME:   {strides = [2, 1, 2]}
//  CHECK-SAME:   memref<?x?x?xf32>, memref<?x?x?xi32>, memref<?x?x?xf32>

// -----

#accesses_0 = [
  affine_map<(i, j, k) -> (j, i)>,
  affine_map<(i, j, k) -> (i, k, i + j)>
]

#trait_0 = {
  indexing_maps = #accesses_0,
  iterator_types = ["parallel", "parallel", "parallel"],
  library_call = "some_external_function_name_1"
}

func @generic(%arg0: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>,
              %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait_0
       ins(%arg0 : memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>)
      outs(%arg1 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>)
      attrs = {foo = 1} {
    ^bb(%0: vector<3x4xi4>, %1: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  }
  return
}
// CHECK-LABEL: func @generic
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [#{{[0-9a-z]*}}, #{{[0-9a-z]*}}],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_1"}
//  CHECK-SAME:      ins({{.*}} : memref<?x?xvector<3x4xi4>, #[[$strided2D]]>)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, #[[$strided3D]]>)
//  CHECK-SAME:     {foo = 1 : i64}

func @generic_with_tensor_input(%arg0: tensor<?x?xvector<3x4xi4>>,
                                %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait_0
       ins(%arg0 : tensor<?x?xvector<3x4xi4>>)
      outs(%arg1 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>)
      attrs = {foo = 1} {
    ^bb(%0: vector<3x4xi4>, %1: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  }
  return
}
// CHECK-LABEL: func @generic_with_tensor_input
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_1"}
//  CHECK-SAME:     ins({{.*}} : tensor<?x?xvector<3x4xi4>>)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, #[[$strided3D]]>)
//  CHECK-SAME:     {foo = 1 : i64}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_without_inputs(%arg0 : memref<?x?x?xf32>) {
  linalg.generic  {indexing_maps = [#map0],
                   iterator_types = ["parallel", "parallel", "parallel"]}
                  outs(%arg0 : memref<?x?x?xf32>) {
   ^bb0(%arg3: f32):  // no predecessors
      %cst = constant 0.000000e+00 : f32
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
      %f0 = constant 0.0 : f32
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

#accesses_2 = [
  affine_map<(i, j, k) -> (j, i)>,
  affine_map<(i, j, k) -> (i, k, i + j)>,
  affine_map<(i, j, k) -> (i, k, i + j)>
]

#trait_2 = {
  indexing_maps = #accesses_2,
  iterator_types = ["parallel", "parallel", "parallel"],
  library_call = "some_external_function_name_1"
}

func @indexed_generic_with_tensor_input_and_output(
    %arg0: tensor<?x?xvector<3x4xi4>>, %arg1: tensor<?x?x?xf32>)
    -> (tensor<?x?x?xf32>) {
  %0 = linalg.indexed_generic #trait_2
       ins(%arg0, %arg1 : tensor<?x?xvector<3x4xi4>>, tensor<?x?x?xf32>)
      outs(%arg1 : tensor<?x?x?xf32>)
      attrs = {foo = 1} {
    ^bb(%i: index, %j: index, %k: index, %0: vector<3x4xi4>, %1: f32, %2: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @indexed_generic_with_tensor_input_and_output
//       CHECK:   linalg.indexed_generic {
//  CHECK-SAME:     indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_1"}
//  CHECK-SAME:      ins({{.*}} : tensor<?x?xvector<3x4xi4>>, tensor<?x?x?xf32>)
//  CHECK-SAME:     outs({{.*}} : tensor<?x?x?xf32>)
//  CHECK-SAME:     {foo = 1 : i64}
//       CHECK:     -> tensor<?x?x?xf32>
//       CHECK:   return {{.*}} : tensor<?x?x?xf32>

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

func @indexed_generic_op_zero_rank(%arg0: tensor<f32>, %arg1 : tensor<3x4xf32>) -> (tensor<3x4xf32>)
{
  %0 = linalg.indexed_generic #trait_broadcast
       ins(%arg0 : tensor<f32>)
      outs(%arg1 : tensor<3x4xf32>) {
    ^bb(%i: index, %j: index, %a: f32, %b: f32) :
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
//       CHECK:    linalg.yield %{{.*}} : f32

func @indexed_generic(%arg0: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>,
                      %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.indexed_generic #trait_3
       ins(%arg0 : memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>)
      outs(%arg1 : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>)
      attrs = {foo = 1} {
    ^bb(%i: index, %j: index, %k: index, %a: vector<3x4xi4>, %b: f32) :
      linalg.yield %b : f32
  }
  return
}
// CHECK-LABEL: func @indexed_generic
//       CHECK:   linalg.indexed_generic {
//  CHECK-SAME:     indexing_maps = [#{{[0-9a-z]*}}, #{{[0-9a-z]*}}],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"],
//  CHECK-SAME:     library_call = "some_external_function_name_2"
//  CHECK-SAME:      ins({{.*}} : memref<?x?xvector<3x4xi4>, #[[$strided2D]]>)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, #[[$strided3D]]>)
//  CHECK-SAME:     {foo = 1 : i64}
//       CHECK:    ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: vector<3x4xi4>, %{{.*}}: f32):
//       CHECK:      linalg.yield %{{.*}} : f32
//       CHECK:    }

// -----

func @reshape_static(%arg0: memref<3x4x5xf32>, %arg1: tensor<3x4x5xf32>, %arg2: tensor<3x?x5xf32>) {
  // Reshapes that collapse and expand back a contiguous buffer.
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k) -> (i, j)>,
                             affine_map<(i, j, k) -> (k)>] :
    memref<3x4x5xf32> into memref<12x5xf32>
  %r0 = linalg.reshape %0 [affine_map<(i, j, k) -> (i, j)>,
                           affine_map<(i, j, k) -> (k)>] :
    memref<12x5xf32> into memref<3x4x5xf32>
  %1 = linalg.reshape %arg0 [affine_map<(i, j, k) -> (i)>,
                             affine_map<(i, j, k) -> (j, k)>] :
    memref<3x4x5xf32> into memref<3x20xf32>
  %r1 = linalg.reshape %1 [affine_map<(i, j, k) -> (i)>,
                           affine_map<(i, j, k) -> (j, k)>] :
    memref<3x20xf32> into memref<3x4x5xf32>
  %2 = linalg.reshape %arg0 [affine_map<(i, j, k) -> (i, j, k)>] :
    memref<3x4x5xf32> into memref<60xf32>
  %r2 = linalg.reshape %2 [affine_map<(i, j, k) -> (i, j, k)>] :
    memref<60xf32> into memref<3x4x5xf32>
  // Reshapes that expand and collapse back a contiguous buffer with some 1's.
  %3 = linalg.reshape %arg0 [affine_map<(i, j, k, l, m) -> (i, j)>,
                             affine_map<(i, j, k, l, m) -> (k)>,
                             affine_map<(i, j, k, l, m) -> (l, m)>] :
    memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  %r3 = linalg.reshape %3 [affine_map<(i, j, k, l, m) -> (i, j)>,
                           affine_map<(i, j, k, l, m) -> (k)>,
                           affine_map<(i, j, k, l, m) -> (l, m)>] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  // Reshapes on tensors.
  %t0 = linalg.tensor_reshape %arg1 [affine_map<(i, j, k, l, m) -> (i, j)>,
                                     affine_map<(i, j, k, l, m) -> (k)>,
                                     affine_map<(i, j, k, l, m) -> (l, m)>] :
    tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>
  %rt0 = linalg.tensor_reshape %t0 [affine_map<(i, j, k, l, m) -> (i, j)>,
                                    affine_map<(i, j, k, l, m) -> (k)>,
                                    affine_map<(i, j, k, l, m) -> (l, m)>] :
    tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>
  %t1 = linalg.tensor_reshape %arg2 [affine_map<(i, j, k, l, m) -> (i, j)>,
                                     affine_map<(i, j, k, l, m) -> (k)>,
                                     affine_map<(i, j, k, l, m) -> (l, m)>] :
    tensor<3x?x5xf32> into tensor<1x3x?x1x5xf32>
  %rt1 = linalg.tensor_reshape %t1 [affine_map<(i, j, k, l, m) -> (i)>,
                                    affine_map<(i, j, k, l, m) -> (j, k)>,
                                    affine_map<(i, j, k, l, m) -> (l, m)>] :
    tensor<1x3x?x1x5xf32> into tensor<1x?x5xf32>
  return
}
// CHECK-LABEL: func @reshape_static
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<12x5xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<12x5xf32> into memref<3x4x5xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD0]], #[[$reshapeD12]]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<3x20xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD0]], #[[$reshapeD12]]]
//  CHECK-SAME:     memref<3x20xf32> into memref<3x4x5xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD012]]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<60xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD012]]]
//  CHECK-SAME:     memref<60xf32> into memref<3x4x5xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshape5D01]], #[[$reshape5D2]], #[[$reshape5D34]]]
//  CHECK-SAME:     memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshape5D01]], #[[$reshape5D2]], #[[$reshape5D34]]]
//  CHECK-SAME:     memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
//
//       CHECK:   linalg.tensor_reshape {{.*}}: tensor<3x4x5xf32> into tensor<1x3x4x1x5xf32>
//       CHECK:   linalg.tensor_reshape {{.*}}: tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>
//       CHECK:   linalg.tensor_reshape {{.*}}: tensor<3x?x5xf32> into tensor<1x3x?x1x5xf32>
//       CHECK:   linalg.tensor_reshape {{.*}}: tensor<1x3x?x1x5xf32> into tensor<1x?x5xf32>

// -----

func @reshape_dynamic(%arg0: memref<?x?x?xf32>,
                      %arg1: memref<?x?x?xf32, offset : 0, strides : [?, ?, 1]>,
                      %arg2: memref<?x?x?xf32, offset : ?, strides : [?, ?, 1]>) {
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k) -> (i, j)>,
                             affine_map<(i, j, k) -> (k)>] :
    memref<?x?x?xf32> into memref<?x?xf32>
  %r0 = linalg.reshape %0 [affine_map<(i, j, k) -> (i, j)>,
                           affine_map<(i, j, k) -> (k)>] :
    memref<?x?xf32> into memref<?x4x?xf32>
  %1 = linalg.reshape %arg1 [affine_map<(i, j, k) -> (i, j)>,
                             affine_map<(i, j, k) -> (k)>] :
    memref<?x?x?xf32, offset : 0, strides : [?, ?, 1]> into
    memref<?x?xf32, offset : 0, strides : [?, 1]>
  %r1 = linalg.reshape %1 [affine_map<(i, j, k) -> (i, j)>,
                           affine_map<(i, j, k) -> (k)>] :
    memref<?x?xf32, offset : 0, strides : [?, 1]> into
    memref<?x4x?xf32, offset : 0, strides : [?, ?, 1]>
  %2 = linalg.reshape %arg2 [affine_map<(i, j, k) -> (i, j)>,
                             affine_map<(i, j, k) -> (k)>] :
    memref<?x?x?xf32, offset : ?, strides : [?, ?, 1]> into
    memref<?x?xf32, offset : ?, strides : [?, 1]>
  %r2 = linalg.reshape %2 [affine_map<(i, j, k) -> (i, j)>,
                           affine_map<(i, j, k) -> (k)>] :
    memref<?x?xf32, offset : ?, strides : [?, 1]> into
    memref<?x4x?xf32, offset : ?, strides : [?, ?, 1]>
  return
}

// CHECK-LABEL: func @reshape
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<?x?x?xf32> into memref<?x?xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<?x?xf32> into memref<?x4x?xf32>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3DOFF0]]> into memref<?x?xf32, #[[$strided2DOFF0]]>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<?x?xf32, #[[$strided2DOFF0]]> into memref<?x4x?xf32, #[[$strided3DOFF0]]>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<?x?x?xf32, #[[$strided3D]]> into memref<?x?xf32, #[[$strided2D]]>
//       CHECK:   linalg.reshape {{.*}} [#[[$reshapeD01]], #[[$reshapeD2]]]
//  CHECK-SAME:     memref<?x?xf32, #[[$strided2D]]> into memref<?x4x?xf32, #[[$strided3D]]>

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

func @tensor_reshape_zero_dim(%arg0 : tensor<1x1xf32>, %arg1 : tensor<f32>) -> (tensor<f32>, tensor<1x1xf32>)
{
  %0 = linalg.tensor_reshape %arg0 [] : tensor<1x1xf32> into tensor<f32>
  %1 = linalg.tensor_reshape %0 [] : tensor<f32> into tensor<1x1xf32>
  return %0, %1 : tensor<f32>, tensor<1x1xf32>
}
// CHECK-LABEL: func @tensor_reshape_zero_dim
//       CHECK:   linalg.tensor_reshape %{{.*}} [] : tensor<1x1xf32> into tensor<f32>
//       CHECK:   linalg.tensor_reshape %{{.*}} [] : tensor<f32> into tensor<1x1xf32>

// -----

func @memref_reshape_zero_dim(%arg0 : memref<1x1xf32>, %arg1 : memref<f32>) -> (memref<f32>, memref<1x1xf32>)
{
  %0 = linalg.reshape %arg0 [] : memref<1x1xf32> into memref<f32>
  %1 = linalg.reshape %0 [] : memref<f32> into memref<1x1xf32>
  return %0, %1 : memref<f32>, memref<1x1xf32>
}
// CHECK-LABEL: func @memref_reshape_zero_dim
//       CHECK:   linalg.reshape %{{.*}} [] : memref<1x1xf32> into memref<f32>
//       CHECK:   linalg.reshape %{{.*}} [] : memref<f32> into memref<1x1xf32>

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

func @legal_collapsing_reshape_dynamic_tensor
  (%arg0: tensor<?x?x?x4x?xf32>) -> tensor<?x?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    tensor<?x?x?x4x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//     CHECK: func @legal_collapsing_reshape_dynamic_tensor
//     CHECK:   linalg.tensor_reshape %{{.+}} [#[[$reshape5D0]], #[[$reshape5D1]], #[[$reshape5D345]]]

// -----

func @legal_collapsing_reshape_dynamic_memref
  (%arg0: memref<?x?x?x4x?xf32>) -> memref<?x?x?xf32>
{
  %0 = linalg.reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    memref<?x?x?x4x?xf32> into memref<?x?x?xf32>
  return %0 : memref<?x?x?xf32>
}
//     CHECK: func @legal_collapsing_reshape_dynamic_memref
//     CHECK:   linalg.reshape %{{.+}} [#[[$reshape5D0]], #[[$reshape5D1]], #[[$reshape5D345]]]

// -----

func @fill_tensor(%arg0 : index, %arg1 : index, %arg2 : f32) -> tensor<?x?xf32> {
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = linalg.fill(%0, %arg2) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK: %{{.+}} = linalg.fill(%{{.+}}, %{{.+}}) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>

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
 %c0 = constant 0 : index
 %c1 = constant 1 : index
 %c4 = constant 4 : index
 %c24 = constant 24 : index
 %c64 = constant 64 : index
 %prod = linalg.tiled_loop (%i) = (%c0) to (%c24) step (%c4)
      ins(%lhs, %rhs : tensor<24x64xi8>, tensor<24x64xi8>)
      outs(%out : tensor<24x64xi8>) {
    %lhs_sub = subtensor %lhs[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>
    %rhs_sub = subtensor %rhs[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>
    %out_sub = subtensor %out[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>

    %sum = linalg.generic #trait_4
        ins(%lhs_sub, %rhs_sub : tensor<?x?xi8>, tensor<?x?xi8>)
        outs(%out_sub : tensor<?x?xi8>) {
      ^bb(%l: i8, %r: i8, %o: i8) :
        %s = addi %l, %r : i8
        linalg.yield %s : i8
      } -> tensor<?x?xi8>

    %sum_sub = subtensor_insert %sum into %out[%i, 0][%c4, %c64][1, 1]
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
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c8 = constant 8 : index
  %X = memref.dim %input_3d, %c0 : tensor<16x24x32xf32>
  %Y = memref.dim %input_3d, %c1 : tensor<16x24x32xf32>
  %Z = memref.dim %input_3d, %c2 : tensor<16x24x32xf32>
  %result = linalg.tiled_loop (%i, %j, %k)
      = (%c0, %c0, %c0) to (%X, %Y, %Z) step (%c2, %c4, %c8)
      ins(%input_3d, %input_2d: tensor<16x24x32xf32>, tensor<16x32xf32>)
      outs( %output: tensor<24xf32>)
      iterators["reduction", "parallel", "reduction"] {
    %sub_3d = subtensor %input_3d[%i, %j, %k][2, 4, 8][1, 1, 1]
      : tensor<16x24x32xf32> to tensor<2x4x8xf32>
    %sub_2d = subtensor %input_2d[%i, %k][2, 8][1, 1]
      : tensor<16x32xf32> to tensor<2x8xf32>
    %sub_1d = subtensor %input_1d[%j] [4] [1]
      : tensor<24xf32> to tensor<4xf32>
    %sub_out = subtensor %output[%j] [4] [1]
      : tensor<24xf32> to tensor<4xf32>
    %acc = linalg.generic #trait_5
      ins(%sub_3d, %sub_2d, %sub_1d
        : tensor<2x4x8xf32>, tensor<2x8xf32>, tensor<4xf32>)
      outs(%sub_out : tensor<4xf32>)  {
    ^bb0(%i3d: f32, %i2d: f32, %i1d: f32, %o: f32):
      %0 = addf %i3d, %i2d : f32
      %1 = addf %0, %i1d : f32
      linalg.yield %1 : f32
    } -> tensor<4xf32>

    %sum_sub = subtensor_insert %acc into %output[%j][%c4][1]
      : tensor<4xf32> into tensor<24xf32>
    linalg.yield %sum_sub : tensor<24xf32>
  }
  return %result : tensor<24xf32>
}
// CHECK-LABEL: func @tiled_loop_reduction
// CHECK: iterators[
