// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// TODO: Re-enable LLVM lowering test.
//
// Test that we can lower all the way to LLVM without crashing, don't check results here.
// DISABLED: mlir-opt %s -o=/dev/null 2>&1

func.func @views(%arg0: index) {
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

func.func @ops(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>,
          %arg1: memref<?xf32, strided<[1], offset: ?>>,
          %arg2: memref<?xf32, strided<[1], offset: ?>>,
          %arg3: memref<f32>) {
  linalg.matmul ins(%arg0, %arg0 : memref<?x?xf32, strided<[?, 1], offset: ?>>,
                                   memref<?x?xf32, strided<[?, 1], offset: ?>>)
               outs(%arg0 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
  linalg.matvec ins(%arg0, %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                                  memref<?xf32, strided<[1], offset: ?>>)
               outs(%arg2: memref<?xf32, strided<[1], offset: ?>>)
  linalg.dot ins(%arg1, %arg2: memref<?xf32, strided<[1], offset: ?>>,
                               memref<?xf32, strided<[1], offset: ?>>)
            outs(%arg3: memref<f32>)
  return
}
// CHECK-LABEL: func @ops(%
// CHECK: linalg.matmul
// CHECK-SAME:   ins(%{{.*}}, %{{.*}} : memref<?x?xf32, strided<[?, 1], offset: ?>>,
// CHECK-SAME:                          memref<?x?xf32, strided<[?, 1], offset: ?>>)
// CHECK-SAME:  outs(%{{.*}} : memref<?x?xf32, strided<[?, 1], offset: ?>>)
// CHECK: linalg.matvec
// CHECK-SAME:   ins(%{{.*}}, %{{.*}}: memref<?x?xf32, strided<[?, 1], offset: ?>>,
// CHECK-SAME:                         memref<?xf32, strided<[1], offset: ?>>)
// CHECK-SAME:  outs(%{{.*}}: memref<?xf32, strided<[1], offset: ?>>)
// CHECK: linalg.dot
// CHECK-SAME:   ins(%{{.*}}, %{{.*}}: memref<?xf32, strided<[1], offset: ?>>,
// CHECK-SAME:                         memref<?xf32, strided<[1], offset: ?>>)
// CHECK-SAME:  outs(%{{.*}}: memref<f32>)

// -----

func.func @fill_view(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: f32) {
  linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<?xf32, strided<[1], offset: ?>>)
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK:  %{{.*}}: memref<?xf32, strided<[1], offset: ?>>, %{{.*}}: f32) {
//       CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<?xf32, strided<[1], offset: ?>>)

// -----

func.func @memref_transpose(%arg0: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
  %0 = memref.transpose %arg0 (i, j, k) -> (k, j, i) : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?x?xf32, strided<[1, ?, ?], offset: ?>>
  return
}
// CHECK-LABEL: func @memref_transpose
//       CHECK:   memref.transpose %{{.*}} ([[i:.*]], [[j:.*]], [[k:.*]]) -> ([[k]], [[j]], [[i]]) :
//  CHECK-SAME:      memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?x?xf32, strided<[1, ?, ?], offset: ?>>

// -----


func.func @fill_view3(%arg0: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, %arg1: f32) {
  linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, %{{.*}}: f32) {
//       CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)

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

func.func @generic(%arg0: memref<?x?xvector<3x4xi4>, strided<[?, 1], offset: ?>>,
              %arg1: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
  %cst = arith.constant 0.0 : f32
  linalg.generic #trait_0
       ins(%arg0, %cst : memref<?x?xvector<3x4xi4>, strided<[?, 1], offset: ?>>, f32)
      outs(%arg1 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)
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
//  CHECK-SAME:      ins({{.*}}, {{.*}} : memref<?x?xvector<3x4xi4>, strided<[?, 1], offset: ?>>, f32)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)
//  CHECK-SAME:     {foo = 1 : i64}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @generic_without_inputs(%arg0 : memref<?x?x?xf32>) {
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

func.func @generic_with_tensor_input_and_output(
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

func.func @generic_with_multiple_tensor_outputs(
    %arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: i32)
    -> (tensor<i32>, tensor<i32>) {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<i32>
  %1 = linalg.fill ins(%arg2 : i32) outs(%0 : tensor<i32>) -> tensor<i32>
  %2 = tensor.empty() : tensor<i32>
  %3 = linalg.fill ins(%arg2 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
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

func.func @generic_op_zero_rank(%arg0: tensor<f32>, %arg1 : tensor<3x4xf32>) -> (tensor<3x4xf32>)
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

func.func @generic_region(%arg0: memref<?x?xvector<3x4xi4>, strided<[?, 1], offset: ?>>,
                     %arg1: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
  linalg.generic #trait_3
       ins(%arg0 : memref<?x?xvector<3x4xi4>, strided<[?, 1], offset: ?>>)
      outs(%arg1 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)
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
//  CHECK-SAME:      ins({{.*}} : memref<?x?xvector<3x4xi4>, strided<[?, 1], offset: ?>>)
//  CHECK-SAME:     outs({{.*}} : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)
//  CHECK-SAME:     attrs = {foo = 1 : i64} {
//       CHECK:  ^{{.*}}(%{{.*}}: vector<3x4xi4>, %{{.*}}: f32):
//       CHECK:    %{{.*}} = linalg.index 0 : index
//       CHECK:    %{{.*}} = linalg.index 1 : index
//       CHECK:    %{{.*}} = linalg.index 2 : index
//       CHECK:    linalg.yield %{{.*}} : f32

// -----


func.func @named_ops(%a3: memref<?x?x?xf32>, %b3: memref<?x?x?xf32>, %c3: memref<?x?x?xf32>,
                %ta3: tensor<?x?x?xf32>, %tb3: tensor<?x?x?xf32>, %tc3: tensor<?x?x?xf32>)
  -> (tensor<?x?x?xf32>)
{
  linalg.batch_matmul ins(%a3, %b3: memref<?x?x?xf32>, memref<?x?x?xf32>)
                     outs(%c3: memref<?x?x?xf32>)
  %res1 = linalg.batch_matmul
                      ins(%ta3, %tb3: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
                     outs(%tc3: tensor<?x?x?xf32>)
                  -> tensor<?x?x?xf32>
  return %res1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @named_ops
//       CHECK:   linalg.batch_matmul
//       CHECK:   linalg.batch_matmul

// -----

func.func @fill_tensor(%arg0 : index, %arg1 : index, %arg2 : f32) -> tensor<?x?xf32> {
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%arg2 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK: %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @mixed_parallel_reduced_results(%arg0 : tensor<?x?x?xf32>,
    %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?x?xf32>, %arg3 : tensor<?x?xf32>) ->
    (tensor<?x?x?xf32>, tensor<?x?xf32>) {
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%arg2, %arg3 : tensor<?x?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32, %b3 : f32):
      %1 = arith.mulf %b0, %b1 : f32
      %2 = arith.addf %1, %b3 : f32
      linalg.yield %1, %2 : f32, f32
  } -> (tensor<?x?x?xf32>, tensor<?x?xf32>)
  return %0#0, %0#1 : tensor<?x?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func @mixed_parallel_reduced_results
//       CHECK:     linalg.generic

// -----

func.func @map_no_inputs(%init: tensor<64xf32>) -> tensor<64xf32> {
   %add = linalg.map
      outs(%init:tensor<64xf32>)
      () {
        %0 = arith.constant 0.0: f32
        linalg.yield %0: f32
      }
  func.return %add : tensor<64xf32>
}
// CHECK-LABEL: func @map_no_inputs
//       CHECK:   linalg.map outs
//  CHECK-NEXT:   () {
//  CHECK-NEXT:     arith.constant
//  CHECK-NEXT:     linalg.yield
//  CHECK-NEXT:   }

// -----

func.func @map_binary(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>,
                      %init: tensor<64xf32>) -> tensor<64xf32> {
   %add = linalg.map
          ins(%lhs, %rhs: tensor<64xf32>, tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            linalg.yield %0: f32
          }
  func.return %add : tensor<64xf32>
}
// CHECK-LABEL: func @map_binary
//       CHECK:   linalg.map { arith.addf } ins
//  CHECK-SAME:   outs

// -----

func.func @map_binary_memref(%lhs: memref<64xf32>, %rhs: memref<64xf32>,
                      %init: memref<64xf32>) {
   linalg.map
      ins(%lhs, %rhs: memref<64xf32>, memref<64xf32>)
      outs(%init:memref<64xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @map_binary_memref
//       CHECK:     linalg.map

// -----

func.func @map_unary(%input: tensor<64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
   %abs = linalg.map
          ins(%input:tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%input_elem: f32) {
            %0 = math.absf %input_elem: f32
            linalg.yield %0: f32
          }
  func.return %abs : tensor<64xf32>
}
// CHECK-LABEL: func @map_unary
//       CHECK:     linalg.map

// -----

func.func @map_unary_memref(%input: memref<64xf32>, %init: memref<64xf32>) {
   linalg.map
      ins(%input:memref<64xf32>)
      outs(%init:memref<64xf32>)
      (%input_elem: f32) {
        %0 = math.absf %input_elem: f32
        linalg.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @map_unary_memref
//       CHECK:     linalg.map

// -----

func.func @reduce(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}
// CHECK-LABEL: func @reduce
//       CHECK:   linalg.reduce { arith.addf } ins
//  CHECK-SAME:   outs
//  CHECK-SAME:   dimensions = [1]

// -----

func.func @reduce_memref(%input: memref<16x32x64xf32>,
                         %init: memref<16x64xf32>) {
  linalg.reduce
      ins(%input:memref<16x32x64xf32>)
      outs(%init:memref<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @reduce
//       CHECK:   linalg.reduce { arith.addf } ins
//  CHECK-SAME:   outs
//  CHECK-SAME:   dimensions = [1]

// -----

func.func @variadic_reduce(%input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xi64>,
    %init2: tensor<16x64xi64>)  -> (tensor<16x64xf32>, tensor<16x64xi64>) {
  %reduce, %reduce2 = linalg.reduce
      ins(%input1, %input2 : tensor<16x32x64xf32>, tensor<16x32x64xi64>)
      outs(%init1, %init2 : tensor<16x64xf32>, tensor<16x64xi64>)
      dimensions = [1]
      (%in1: f32, %in2: i64, %out1: f32, %out2: i64) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addi %in2, %out2: i64
        linalg.yield %0, %1: f32, i64
      }
  func.return %reduce, %reduce2 : tensor<16x64xf32>, tensor<16x64xi64>
}
// CHECK-LABEL: func @variadic_reduce
//       CHECK:     linalg.reduce
//   CHECK-NOT:     { arith.addf

// -----

func.func @variadic_reduce_memref(%input1: memref<16x32x64xf32>,
    %init1: memref<16x64xf32>, %input2: memref<16x32x64xi64>,
    %init2: memref<16x64xi64>) {
  linalg.reduce
      ins(%input1, %input2 : memref<16x32x64xf32>, memref<16x32x64xi64>)
      outs(%init1, %init2 : memref<16x64xf32>, memref<16x64xi64>)
      dimensions = [1]
      (%in1: f32, %in2: i64, %out1: f32, %out2: i64) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addi %in2, %out2: i64
        linalg.yield %0, %1: f32, i64
      }
  func.return
}
//   CHECK-LABEL: func @variadic_reduce_memref
//       CHECK:     linalg.reduce
//   CHECK-NOT:     { arith.addf

// -----

func.func @transpose(%input: tensor<16x32x64xf32>,
                     %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 2, 0]
  func.return %transpose : tensor<32x64x16xf32>
}
// CHECK-LABEL: func @transpose
//      CHECK:    linalg.transpose ins
// CHECK-SAME:    outs
// CHECK-SAME:    permutation

// -----

func.func @transpose_memref(%input: memref<16x32x64xf32>,
                            %init: memref<32x64x16xf32>) {
  linalg.transpose
      ins(%input:memref<16x32x64xf32>)
      outs(%init:memref<32x64x16xf32>)
      permutation = [1, 2, 0]
  func.return
}
// CHECK-LABEL: func @transpose_memref

// -----

func.func @broadcast_static_sizes(%input: tensor<8x32xf32>,
                            %init: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %bcast = linalg.broadcast
      ins(%input:tensor<8x32xf32>)
      outs(%init:tensor<8x16x32xf32>)
      dimensions = [1]
  func.return %bcast : tensor<8x16x32xf32>
}
// CHECK-LABEL: func @broadcast_static_sizes
//      CHECK:    linalg.broadcast ins
// CHECK-SAME:    outs
// CHECK-SAME:    dimensions

// -----

func.func @broadcast_with_dynamic_sizes(
              %input: tensor<8x?xf32>, %init: tensor<8x16x?xf32>)
              -> tensor<8x16x?xf32> {
  %bcast = linalg.broadcast
      ins(%input:tensor<8x?xf32>)
      outs(%init:tensor<8x16x?xf32>)
      dimensions = [1]
  func.return %bcast : tensor<8x16x?xf32>
}
// CHECK-LABEL: func @broadcast_with_dynamic_sizes
//      CHECK:    linalg.broadcast ins
// CHECK-SAME:    outs
// CHECK-SAME:    dimensions

// -----

func.func @broadcast_memref(%input: memref<8x32xf32>,
                            %init: memref<8x16x32xf32>) {
  linalg.broadcast
      ins(%input:memref<8x32xf32>)
      outs(%init:memref<8x16x32xf32>)
      dimensions = [1]
  func.return
}

// CHECK-LABEL: func @broadcast_memref
//      CHECK:    linalg.broadcast ins
// CHECK-SAME:    outs
// CHECK-SAME:    dimensions

// -----

func.func @map_arith_with_attr(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>,
                      %init: tensor<64xf32>) -> tensor<64xf32> {
  %add = linalg.map
          ins(%lhs, %rhs: tensor<64xf32>, tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem fastmath<fast> : f32
            linalg.yield %0: f32
          }
  func.return %add : tensor<64xf32>
}

// CHECK-LABEL: func @map_arith_with_attr
// CHECK-NEXT:    %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:    { arith.addf {fastmath = #arith.fastmath<fast>} }
// CHECK-SAME:    ins
// CHECK-SAME:    outs
// CHECK-NEXT:    return %[[MAPPED]] : tensor<64xf32>

// -----

func.func @reduce_arith_with_attr(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in fastmath<fast> : f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}
// CHECK-LABEL: func @reduce_arith_with_attr
// CHECK-NEXT:    %[[REDUCED:.*]] = linalg.reduce
// CHECK-SAME:    { arith.addf {fastmath = #arith.fastmath<fast>} }
// CHECK-SAME:    ins
// CHECK-SAME:    outs
// CHECK-SAME:    dimensions = [1]
// CHECK-NEXT:    return %[[REDUCED]] : tensor<16x64xf32>

// -----

func.func @softmax(%arg0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %0 = tensor.empty() : tensor<2x16x32xf32>
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}
// CHECK:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK:        %[[D1:.+]] = linalg.softmax dimension(2) ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D0]] :
// CHECK-SAME:     tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK:        return %[[D1]] : tensor<2x16x32xf32>
// CHECK:      }

// -----

func.func @winograd(%arg0: tensor<2x6x6x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %arg3: tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32> {
  %0 = tensor.empty() : tensor<6x6x5x2xf32>
  %1 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%0 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  %2 = tensor.empty() : tensor<6x6x1x1x2x5xf32>
  %3 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x6x6x5xf32>) outs(%2 : tensor<6x6x1x1x2x5xf32>) -> tensor<6x6x1x1x2x5xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<6x6x5x2xf32> into tensor<36x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %3 [[0, 1], [2, 3, 4], [5]] : tensor<6x6x1x1x2x5xf32> into tensor<36x2x5xf32>
  %4 = tensor.empty() : tensor<36x2x2xf32>
  %5 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<36x2x5xf32>, tensor<36x5x2xf32>) outs(%4 : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
  %expanded = tensor.expand_shape %5 [[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 1, 1, 2, 2] : tensor<36x2x2xf32> into tensor<6x6x1x1x2x2xf32>
  %6 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<6x6x1x1x2x2xf32>) outs(%arg3 : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
  return %6 : tensor<2x4x4x2xf32>
}

// CHECK-LABEL: func @winograd
// CHECK:         linalg.winograd_filter_transform m(4) r(3)
// CHECK:         linalg.winograd_input_transform m(4) r(3)
// CHECK:         linalg.winograd_output_transform m(4) r(3)

// -----

func.func @winograd_filter_dyn(%arg0: tensor<?x3x3x?xf32>, %arg1: tensor<6x6x?x?xf32>) -> tensor<6x6x?x?xf32> {
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<?x3x3x?xf32>) outs(%arg1 : tensor<6x6x?x?xf32>) -> tensor<6x6x?x?xf32>
  return %0 : tensor<6x6x?x?xf32>
}

// CHECK-LABEL: func @winograd_filter_dyn
// CHECK:         linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<?x3x3x?xf32>) outs(%arg1 : tensor<6x6x?x?xf32>) -> tensor<6x6x?x?xf32>

// -----

func.func @winograd_input_dyn(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<6x6x?x?x?x?xf32>) -> tensor<6x6x?x?x?x?xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<?x?x?x?xf32>) outs(%arg1 : tensor<6x6x?x?x?x?xf32>) -> tensor<6x6x?x?x?x?xf32>
  return %0 : tensor<6x6x?x?x?x?xf32>
}

// CHECK-LABEL: func @winograd_input_dyn
// CHECK:         linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<?x?x?x?xf32>) outs(%arg1 : tensor<6x6x?x?x?x?xf32>) -> tensor<6x6x?x?x?x?xf32>

// -----

func.func @winograd_output_dyn(%arg0: tensor<6x6x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x?x?x?x?xf32>) outs(%arg1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func @winograd_output_dyn
// CHECK:         linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x?x?x?x?xf32>) outs(%arg1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

// -----

func.func @conv2d_channel_first_q(%img: tensor<100x3x224x224xi32>, %filt: tensor<64x3x5x5xi32>, %a: i32, %b: i32) -> tensor<100x64x220x220xi32> {
  %init = arith.constant dense<0> : tensor<100x64x220x220xi32>
  %1 = linalg.conv_2d_nchw_fchw_q  {dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>}
    ins(%img, %filt, %a, %b : tensor<100x3x224x224xi32>, tensor<64x3x5x5xi32>, i32, i32)
    outs(%init : tensor<100x64x220x220xi32>) -> tensor<100x64x220x220xi32>
  return %1 : tensor<100x64x220x220xi32>
}

// CHECK-LABEL: func @conv2d_channel_first_q(
// CHECK:   %[[arg0:[a-zA-z0-9]*]]: tensor<100x3x224x224xi32>, %[[arg1:[a-zA-z0-9]*]]: tensor<64x3x5x5xi32>, %[[arg2:[a-zA-z0-9]*]]: i32, %[[arg3:[a-zA-z0-9]*]]: i32)
// CHECK:         linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]] : tensor<100x3x224x224xi32>, tensor<64x3x5x5xi32>, i32, i32) outs(%{{.*}} : tensor<100x64x220x220xi32>) -> tensor<100x64x220x220xi32>

// -----

func.func @conv2d_channel_first_q_promote(%img: tensor<100x3x224x224xi8>, %filt: tensor<64x3x5x5xi8>, %a: i8, %b: i8) -> tensor<100x64x220x220xi32> {
  %init = arith.constant dense<0> : tensor<100x64x220x220xi32>
  %1 = linalg.conv_2d_nchw_fchw_q  {dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>}
    ins(%img, %filt, %a, %b : tensor<100x3x224x224xi8>, tensor<64x3x5x5xi8>, i8, i8)
    outs(%init : tensor<100x64x220x220xi32>) -> tensor<100x64x220x220xi32>
  return %1 : tensor<100x64x220x220xi32>
}

// CHECK-LABEL: func @conv2d_channel_first_q_promote(
// CHECK:   %[[arg0:[a-zA-z0-9]*]]: tensor<100x3x224x224xi8>, %[[arg1:[a-zA-z0-9]*]]: tensor<64x3x5x5xi8>, %[[arg2:[a-zA-z0-9]*]]: i8, %[[arg3:[a-zA-z0-9]*]]: i8)
// CHECK:         linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]] : tensor<100x3x224x224xi8>, tensor<64x3x5x5xi8>, i8, i8) outs(%{{.*}} : tensor<100x64x220x220xi32>) -> tensor<100x64x220x220xi32>
