// RUN: mlir-opt %s -test-affine-reify-value-bounds="reify-to-func-args" \
// RUN:     -verify-diagnostics -split-input-file | FileCheck %s

// RUN: mlir-opt %s -test-affine-reify-value-bounds="reify-to-func-args use-arith-ops" \
// RUN:     -verify-diagnostics -split-input-file | FileCheck %s --check-prefix=CHECK-ARITH

// CHECK-LABEL: func @reify_through_chain(
//  CHECK-SAME:     %[[sz0:.*]]: index, %[[sz2:.*]]: index
//       CHECK:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   return %[[sz0]], %[[c10]], %[[sz2]]

// CHECK-ARITH-LABEL: func @reify_through_chain(
//  CHECK-ARITH-SAME:     %[[sz0:.*]]: index, %[[sz2:.*]]: index
//       CHECK-ARITH:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK-ARITH:   return %[[sz0]], %[[c10]], %[[sz2]]
func.func @reify_through_chain(%sz0: index, %sz2: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %0 = tensor.empty(%sz0, %sz2) : tensor<?x10x?xf32>
  %1 = tensor.cast %0 : tensor<?x10x?xf32> to tensor<?x?x?xf32>
  %pos = arith.constant 0 : index
  %f = arith.constant 0.0 : f32
  %2 = tensor.insert %f into %1[%pos, %pos, %pos] : tensor<?x?x?xf32>
  %3 = tensor.dim %2, %c2 : tensor<?x?x?xf32>

  %4 = "test.reify_bound"(%2) {dim = 0} : (tensor<?x?x?xf32>) -> (index)
  %5 = "test.reify_bound"(%2) {dim = 1} : (tensor<?x?x?xf32>) -> (index)
  %6 = "test.reify_bound"(%3) : (index) -> (index)

  return %4, %5, %6 : index, index, index
}

// -----

// CHECK-LABEL: func @reify_slice_bound(
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   "test.some_use"(%[[c5]])
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   "test.some_use"(%[[c5]])
func.func @reify_slice_bound(%t: tensor<?x?xi32>, %idx: index, %ub: index, %f: f32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %ub step %c4 {
    %sz = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%iv)[%ub]
    %slice = tensor.extract_slice %t[%idx, %iv] [1, %sz] [1, 1] : tensor<?x?xi32> to tensor<1x?xi32>
    %filled = linalg.fill ins(%f : f32) outs(%slice : tensor<1x?xi32>) -> tensor<1x?xi32>

    %bound = "test.reify_bound"(%filled) {dim = 1, type = "UB"} : (tensor<1x?xi32>) -> (index)
    "test.some_use"(%bound) : (index) -> ()

    %bound_const = "test.reify_bound"(%filled) {dim = 1, type = "UB", constant} : (tensor<1x?xi32>) -> (index)
    "test.some_use"(%bound_const) : (index) -> ()
  }
  return
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 - s1 + 1)>
// CHECK-LABEL: func @scf_for(
//  CHECK-SAME:     %[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index
//       CHECK:   %[[bound:.*]] = affine.apply #[[$map]]()[%[[ub]], %[[lb]]]
//       CHECK:   "test.some_use"(%[[bound]])
func.func @scf_for(%lb: index, %ub: index, %step: index) {
  scf.for %iv = %lb to %ub step %step {
    %0 = affine.apply affine_map<(d0)[s0] -> (-d0 + s0)>(%iv)[%ub]
    %bound = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
    "test.some_use"(%bound) : (index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @reify_slice_bound2(
func.func @reify_slice_bound2(%lb0: index, %ub0: index, %step0: index,
                              %ub2: index, %t1: tensor<1x?xi8>,
                              %t2: tensor<?x?xi8>, %t3: tensor<1x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %iv0 = %lb0 to %ub0 step %step0 {
    // CHECK: %[[c129:.*]] = arith.constant 129 : index
    // CHECK: "test.some_use"(%[[c129]])
    %ub1 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%iv0)[%ub0]
    %ub1_ub = "test.reify_bound"(%ub1) {type = "UB"} : (index) -> (index)
    "test.some_use"(%ub1_ub) : (index) -> ()

    // CHECK: %[[c129:.*]] = arith.constant 129 : index
    // CHECK: "test.some_use"(%[[c129]])
    %lb1 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 32)>()[%ub1]
    %lb1_ub = "test.reify_bound"(%lb1) {type = "UB"} : (index) -> (index)
    "test.some_use"(%lb1_ub) : (index) -> ()

    // CHECK: %[[c129:.*]] = arith.constant 129 : index
    // CHECK: "test.some_use"(%[[c129]])
    %lb1_ub_const = "test.reify_bound"(%lb1) {type = "UB", constant} : (index) -> (index)
    "test.some_use"(%lb1_ub_const) : (index) -> ()

    scf.for %iv1 = %lb1 to %ub1 step %c32 {
      // CHECK: %[[c32:.*]] = arith.constant 32 : index
      // CHECK: "test.some_use"(%[[c32]])
      %sz = affine.apply affine_map<(d0)[s0] -> (-d0 + s0)>(%iv1)[%ub1]
      %sz_ub = "test.reify_bound"(%sz) {type = "UB"} : (index) -> (index)
      "test.some_use"(%sz_ub) : (index) -> ()

      scf.for %iv2 = %c0 to %ub2 step %c1 {
        %slice1 = tensor.extract_slice %t1[0, %iv2] [1, 1] [1, 1] : tensor<1x?xi8> to tensor<1x1xi8>
        %slice2 = tensor.extract_slice %t2[%iv2, 0] [1, %sz] [1, 1] : tensor<?x?xi8> to tensor<1x?xi8>
        %slice3 = tensor.extract_slice %t3[0, 0] [1, %sz] [1, 1] : tensor<1x?xi32> to tensor<1x?xi32>
        %matmul = linalg.matmul ins(%slice1, %slice2 : tensor<1x1xi8>, tensor<1x?xi8>) outs(%slice3 : tensor<1x?xi32>) -> tensor<1x?xi32>

        // CHECK: %[[c32:.*]] = arith.constant 32 : index
        // CHECK: "test.some_use"(%[[c32]])
        %matmul_ub = "test.reify_bound"(%matmul) {dim = 1, type = "UB"} : (tensor<1x?xi32>) -> (index)
        "test.some_use"(%matmul_ub) : (index) -> ()

        // CHECK: %[[c32:.*]] = arith.constant 32 : index
        // CHECK: "test.some_use"(%[[c32]])
        %matmul_ub_const = "test.reify_bound"(%matmul) {dim = 1, type = "UB", constant} : (tensor<1x?xi32>) -> (index)
        "test.some_use"(%matmul_ub_const) : (index) -> ()
      }
    }
  }
  return
}
