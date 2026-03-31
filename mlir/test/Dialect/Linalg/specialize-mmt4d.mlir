// RUN: mlir-opt %s -linalg-specialize-generic-ops | FileCheck %s

// CHECK-LABEL: @generic_to_mmt4d
// CHECK: linalg.mmt4d
func.func @generic_to_mmt4d(
    %A : tensor<?x?x?x?xf32>,
    %B : tensor<?x?x?x?xf32>,
    %C : tensor<?x?x?x?xf32>
) -> tensor<?x?x?x?xf32> {

  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k, m0, n0, k0) -> (m, k, m0, k0)>,
      affine_map<(m, n, k, m0, n0, k0) -> (n, k, n0, k0)>,
      affine_map<(m, n, k, m0, n0, k0) -> (m, n, m0, n0)>
    ],
    iterator_types = ["parallel", "parallel", "reduction",
                      "parallel", "parallel", "reduction"]
  }
  ins(%A, %B : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  outs(%C : tensor<?x?x?x?xf32>) {
  ^bb0(%a : f32, %b : f32, %c : f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
  } -> tensor<?x?x?x?xf32>

  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @generic_to_mmt4d_transposed_inner
// CHECK: linalg.mmt4d
func.func @generic_to_mmt4d_transposed_inner(
    %A : tensor<?x?x?x?xf32>,
    %B : tensor<?x?x?x?xf32>,
    %C : tensor<?x?x?x?xf32>
) -> tensor<?x?x?x?xf32> {

  %0 = linalg.generic {
    indexing_maps = [
      // Inner dims swapped (m0,k0) to (k0,m0)
      affine_map<(m, n, k, m0, n0, k0) -> (m, k, k0, m0)>,
      affine_map<(m, n, k, m0, n0, k0) -> (n, k, k0, n0)>,
      affine_map<(m, n, k, m0, n0, k0) -> (m, n, m0, n0)>
    ],
    iterator_types = ["parallel", "parallel", "reduction",
                      "parallel", "parallel", "reduction"]
  }
  ins(%A, %B : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  outs(%C : tensor<?x?x?x?xf32>) {
  ^bb0(%a : f32, %b : f32, %c : f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
  } -> tensor<?x?x?x?xf32>

  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @no_mmt4d_bad_map
// CHECK-NOT: linalg.mmt4d
func.func @no_mmt4d_bad_map(
    %A : tensor<?x?x?x?xf32>,
    %B : tensor<?x?x?x?xf32>,
    %C : tensor<?x?x?x?xf32>
) -> tensor<?x?x?x?xf32> {

  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k, m0, n0, k0) -> (k, n, k0, n0)>, // bad map
      affine_map<(m, n, k, m0, n0, k0) -> (n, k, n0, k0)>,
      affine_map<(m, n, k, m0, n0, k0) -> (m, n, m0, n0)>
    ],
    iterator_types = ["parallel", "parallel", "reduction",
                      "parallel", "parallel", "reduction"]
  }
  ins(%A, %B : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  outs(%C : tensor<?x?x?x?xf32>) {
  ^bb0(%a : f32, %b : f32, %c : f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
  } -> tensor<?x?x?x?xf32>

  return %0 : tensor<?x?x?x?xf32>
}
