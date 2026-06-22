// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @specialize_add_int(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.addi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_add_int
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>,  %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_sub_int(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.subi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_sub_int
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>,  %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.sub ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_mul_int(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.muli %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_mul_int
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>,  %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.mul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_div_int(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.divsi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_div_int
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>,  %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_div_unsigned_int(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.divui %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_div_unsigned_int
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>,  %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div_unsigned ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_max_int(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.maxsi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_max_int
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>,  %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.max ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_min_int(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.minsi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_min_int
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>,  %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.min ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_add_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_add_float
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_sub_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.subf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_sub_float
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.sub ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_mul_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_mul_float
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.mul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_div_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.divf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_div_float
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_max_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.maximumf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_max_float
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.max ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_min_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.minimumf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_min_float
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.min ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_powf_float(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = math.powf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_powf_float
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.powf ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_add_complex(%arg0: tensor<?x?xcomplex<f32>>, %arg1: tensor<?x?xcomplex<f32>>, %arg2: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%arg2 : tensor<?x?xcomplex<f32>>) {
  ^bb0(%in: complex<f32>, %in_0: complex<f32>, %out: complex<f32>):
    %1 = complex.add %in, %in_0 : complex<f32>
    linalg.yield %1 : complex<f32>
  } -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}
// CHECK-LABEL: specialize_add_complex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xcomplex<f32>>, %[[ARG1:.+]]: tensor<?x?xcomplex<f32>>,  %[[ARG2:.+]]: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%[[ARG2]] : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>

func.func @specialize_sub_complex(%arg0: tensor<?x?xcomplex<f32>>, %arg1: tensor<?x?xcomplex<f32>>, %arg2: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%arg2 : tensor<?x?xcomplex<f32>>) {
  ^bb0(%in: complex<f32>, %in_0: complex<f32>, %out: complex<f32>):
    %1 = complex.sub %in, %in_0 : complex<f32>
    linalg.yield %1 : complex<f32>
  } -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}
// CHECK-LABEL: specialize_sub_complex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xcomplex<f32>>, %[[ARG1:.+]]: tensor<?x?xcomplex<f32>>,  %[[ARG2:.+]]: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
// CHECK-NOT: linalg.generic
// CHECK: linalg.sub ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%[[ARG2]] : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>

func.func @specialize_mul_complex(%arg0: tensor<?x?xcomplex<f32>>, %arg1: tensor<?x?xcomplex<f32>>, %arg2: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%arg2 : tensor<?x?xcomplex<f32>>) {
  ^bb0(%in: complex<f32>, %in_0: complex<f32>, %out: complex<f32>):
    %1 = complex.mul %in, %in_0 : complex<f32>
    linalg.yield %1 : complex<f32>
  } -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}
// CHECK-LABEL: specialize_mul_complex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xcomplex<f32>>, %[[ARG1:.+]]: tensor<?x?xcomplex<f32>>,  %[[ARG2:.+]]: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
// CHECK-NOT: linalg.generic
// CHECK: linalg.mul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%[[ARG2]] : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>

func.func @specialize_div_complex(%arg0: tensor<?x?xcomplex<f32>>, %arg1: tensor<?x?xcomplex<f32>>, %arg2: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%arg2 : tensor<?x?xcomplex<f32>>) {
  ^bb0(%in: complex<f32>, %in_0: complex<f32>, %out: complex<f32>):
    %1 = complex.div %in, %in_0 : complex<f32>
    linalg.yield %1 : complex<f32>
  } -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}
// CHECK-LABEL: specialize_div_complex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xcomplex<f32>>, %[[ARG1:.+]]: tensor<?x?xcomplex<f32>>,  %[[ARG2:.+]]: tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xcomplex<f32>>, tensor<?x?xcomplex<f32>>) outs(%[[ARG2]] : tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>

func.func @specialize_add_bool(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi1>, %arg2: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi1>, tensor<?x?xi1>) outs(%arg2 : tensor<?x?xi1>) {
  ^bb0(%in: i1, %in_0: i1, %out: i1):
    %1 = arith.ori %in, %in_0 : i1
    linalg.yield %1 : i1
  } -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}
// CHECK-LABEL: specialize_add_bool
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi1>, %[[ARG1:.+]]: tensor<?x?xi1>,  %[[ARG2:.+]]: tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi1>, tensor<?x?xi1>) outs(%[[ARG2]] : tensor<?x?xi1>) -> tensor<?x?xi1>

func.func @specialize_mul_bool(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi1>, %arg2: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi1>, tensor<?x?xi1>) outs(%arg2 : tensor<?x?xi1>) {
  ^bb0(%in: i1, %in_0: i1, %out: i1):
    %1 = arith.andi %in, %in_0 : i1
    linalg.yield %1 : i1
  } -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}
// CHECK-LABEL: specialize_mul_bool
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi1>, %[[ARG1:.+]]: tensor<?x?xi1>,  %[[ARG2:.+]]: tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK-NOT: linalg.generic
// CHECK: linalg.mul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi1>, tensor<?x?xi1>) outs(%[[ARG2]] : tensor<?x?xi1>) -> tensor<?x?xi1>

func.func @specialize_sub_swapped_operands(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.subf %in_0, %in : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_sub_swapped_operands
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.sub ins(%[[ARG1]], %[[ARG0]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
