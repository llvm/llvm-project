// RUN: mlir-opt --transform-interpreter --canonicalize --split-input-file %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tile_sizes, %chunk_sizes = transform.structured.continuous_tile_sizes %0 { dimension = 0, target_size = 9 } : (!transform.any_op) -> !transform.any_op
    %linalg_splits = transform.structured.split %0 after %chunk_sizes { dimension = 0, multiway } : !transform.any_op, !transform.any_op
    transform.foreach %linalg_splits, %tile_sizes : !transform.any_op, !transform.any_op {
    ^bb1(%linalg_split: !transform.any_op, %tile_size: !transform.any_op):
      %tiled_linalg_split, %dim0_loop = transform.structured.tile_using_for %linalg_split tile_sizes [%tile_size] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
    transform.yield
  }
}

func.func @continuous_tile_linalg_matmul(
  %arg0: tensor<25x34xf32>, %arg1: tensor<34x25xf32>, %arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<25x34xf32>, tensor<34x25xf32>)
                     outs(%arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32>

  return %0 : tensor<25x25xf32>
}

// CHECK-LABEL: @continuous_tile_linalg_matmul
// CHECK-SAME:  (%[[IN1:.+]]: tensor<25x34xf32>, %[[IN2:.+]]: tensor<34x25xf32>, %[[OUT:.+]]: tensor<25x25xf32>) -> tensor<25x25xf32> {
// CHECK:         %[[C18:.+]] = arith.constant 18 : index
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[C9:.+]] = arith.constant 9 : index
// CHECK:         %[[XSIN18:.+]] = tensor.extract_slice %[[IN1]][0, 0] [18, 34] [1, 1] : tensor<25x34xf32> to tensor<18x34xf32>
// CHECK:         %[[XSOUT18:.+]] = tensor.extract_slice %[[OUT]][0, 0] [18, 25] [1, 1] : tensor<25x25xf32> to tensor<18x25xf32>
// CHECK:         %[[R0:.+]] = scf.for %[[IDX:.+]] = %[[C0]] to %[[C18]] step %[[C9]] iter_args(%[[XSOUT18ARG:.+]] = %[[XSOUT18]]) -> (tensor<18x25xf32>) {
// CHECK:           %[[XSIN19:.+]] = tensor.extract_slice %[[XSIN18]][%[[IDX]], 0] [9, 34] [1, 1] : tensor<18x34xf32> to tensor<9x34xf32>
// CHECK:           %[[XSOUT9:.+]] = tensor.extract_slice %[[XSOUT18ARG]][%[[IDX]], 0] [9, 25] [1, 1] : tensor<18x25xf32> to tensor<9x25xf32>
// CHECK:           %[[MATMUL:.+]] = linalg.matmul ins(%[[XSIN19]], %[[IN2]] : tensor<9x34xf32>, tensor<34x25xf32>) outs(%[[XSOUT9]] : tensor<9x25xf32>) -> tensor<9x25xf32>
// CHECK:           %[[INS9:.+]] = tensor.insert_slice %[[MATMUL]] into %[[XSOUT18ARG]][%[[IDX]], 0] [9, 25] [1, 1] : tensor<9x25xf32> into tensor<18x25xf32>
// CHECK:           scf.yield %[[INS9]] : tensor<18x25xf32>
// CHECK:         }
// CHECK:         %[[INS:.+]] = tensor.insert_slice %[[R0]] into %[[OUT]][0, 0] [18, 25] [1, 1] : tensor<18x25xf32> into tensor<25x25xf32>
// CHECK:         %[[XS1:.+]] = tensor.extract_slice %[[IN1]][18, 0] [7, 34] [1, 1] : tensor<25x34xf32> to tensor<7x34xf32>
// CHECK:         %[[XS2:.+]] = tensor.extract_slice %[[INS]][18, 0] [7, 25] [1, 1] : tensor<25x25xf32> to tensor<7x25xf32>
// CHECK:         %[[XS3:.+]] = tensor.extract_slice %[[XS1]][0, 0] [4, 34] [1, 1] : tensor<7x34xf32> to tensor<4x34xf32>
// CHECK:         %[[XS4:.+]] = tensor.extract_slice %[[XS2]][0, 0] [4, 25] [1, 1] : tensor<7x25xf32> to tensor<4x25xf32>
// CHECK:         %[[R1:.+]] = linalg.matmul ins(%[[XS3]], %[[IN2]] : tensor<4x34xf32>, tensor<34x25xf32>) outs(%[[XS4]] : tensor<4x25xf32>) -> tensor<4x25xf32>
// CHECK:         %[[INS5:.+]] = tensor.insert_slice %[[R1]] into %[[XS2]][0, 0] [4, 25] [1, 1] : tensor<4x25xf32> into tensor<7x25xf32>
// CHECK:         %[[XS6:.+]] = tensor.extract_slice %[[XS1]][4, 0] [3, 34] [1, 1] : tensor<7x34xf32> to tensor<3x34xf32>
// CHECK:         %[[XS7:.+]] = tensor.extract_slice %[[INS5]][4, 0] [3, 25] [1, 1] : tensor<7x25xf32> to tensor<3x25xf32>
// CHECK:         %[[XS8:.+]] = tensor.extract_slice %[[XS6]][0, 0] [2, 34] [1, 1] : tensor<3x34xf32> to tensor<2x34xf32>
// CHECK:         %[[XS9:.+]] = tensor.extract_slice %[[XS7]][0, 0] [2, 25] [1, 1] : tensor<3x25xf32> to tensor<2x25xf32>
// CHECK:         %[[R2:.+]] = linalg.matmul ins(%[[XS8]], %[[IN2]] : tensor<2x34xf32>, tensor<34x25xf32>) outs(%[[XS9]] : tensor<2x25xf32>) -> tensor<2x25xf32>
// CHECK:         %[[INS10:.+]] = tensor.insert_slice %[[R2]] into %[[XS7]][0, 0] [2, 25] [1, 1] : tensor<2x25xf32> into tensor<3x25xf32>
// CHECK:         %[[XS11:.+]] = tensor.extract_slice %[[XS6]][2, 0] [1, 34] [1, 1] : tensor<3x34xf32> to tensor<1x34xf32>
// CHECK:         %[[XS12:.+]] = tensor.extract_slice %[[INS10]][2, 0] [1, 25] [1, 1] : tensor<3x25xf32> to tensor<1x25xf32>
// CHECK:         %[[R3:.+]] = linalg.matmul ins(%[[XS11]], %[[IN2]] : tensor<1x34xf32>, tensor<34x25xf32>) outs(%[[XS12]] : tensor<1x25xf32>) -> tensor<1x25xf32>
// CHECK:         %[[INS13:.+]] = tensor.insert_slice %[[R3]] into %[[INS10]][2, 0] [1, 25] [1, 1] : tensor<1x25xf32> into tensor<3x25xf32>
// CHECK:         %[[INS14:.+]] = tensor.insert_slice %[[INS13]] into %[[INS5]][4, 0] [3, 25] [1, 1] : tensor<3x25xf32> into tensor<7x25xf32>
// CHECK:         %[[INS15:.+]] = tensor.insert_slice %[[INS14]] into %[[INS]][18, 0] [7, 25] [1, 1] : tensor<7x25xf32> into tensor<25x25xf32>
// CHECK:         return %[[INS15]] : tensor<25x25xf32>

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tile_sizes, %chunk_sizes = transform.structured.continuous_tile_sizes %0 { dimension = 0, target_size = 9 } : (!transform.any_op) -> !transform.param<i64>
    %linalg_splits = transform.structured.split %0 after %chunk_sizes { dimension = 0, multiway } : !transform.any_op, !transform.param<i64>
    transform.foreach %linalg_splits, %tile_sizes : !transform.any_op, !transform.param<i64> {
    ^bb1(%linalg_split: !transform.any_op, %tile_size: !transform.param<i64>):
      %tiled_linalg_split, %dim0_loop = transform.structured.tile_using_for %linalg_split tile_sizes [%tile_size] : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
    transform.yield
  }
}

func.func @continuous_tile_static_linalg_matmul(
  %arg0: tensor<25x34xf32>, %arg1: tensor<34x25xf32>, %arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<25x34xf32>, tensor<34x25xf32>)
                     outs(%arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32>

  return %0 : tensor<25x25xf32>
}

// CHECK-LABEL: @continuous_tile_static_linalg_matmul
// CHECK-SAME:  (%[[IN1:.+]]: tensor<25x34xf32>, %[[IN2:.+]]: tensor<34x25xf32>, %[[OUT:.+]]: tensor<25x25xf32>) -> tensor<25x25xf32> {
// CHECK:         %[[C9:.+]] = arith.constant 9 : index
// CHECK:         %[[C18:.+]] = arith.constant 18 : index
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[XSIN18:.+]] = tensor.extract_slice %[[IN1]][0, 0] [18, 34] [1, 1] : tensor<25x34xf32> to tensor<18x34xf32>
// CHECK:         %[[XSOUT18:.+]] = tensor.extract_slice %[[OUT]][0, 0] [18, 25] [1, 1] : tensor<25x25xf32> to tensor<18x25xf32>
// CHECK:         %[[R0:.+]] = scf.for %[[IDX:.+]] = %[[C0]] to %[[C18]] step %[[C9]] iter_args(%[[XSOUT18ARG:.+]] = %[[XSOUT18]]) -> (tensor<18x25xf32>) {
// CHECK:           %[[XSIN19:.+]] = tensor.extract_slice %[[XSIN18]][%[[IDX]], 0] [9, 34] [1, 1] : tensor<18x34xf32> to tensor<9x34xf32>
// CHECK:           %[[XSOUT9:.+]] = tensor.extract_slice %[[XSOUT18ARG]][%[[IDX]], 0] [9, 25] [1, 1] : tensor<18x25xf32> to tensor<9x25xf32>
// CHECK:           %[[MATMUL:.+]] = linalg.matmul ins(%[[XSIN19]], %[[IN2]] : tensor<9x34xf32>, tensor<34x25xf32>) outs(%[[XSOUT9]] : tensor<9x25xf32>) -> tensor<9x25xf32>
// CHECK:           %[[INS9:.+]] = tensor.insert_slice %[[MATMUL]] into %[[XSOUT18ARG]][%[[IDX]], 0] [9, 25] [1, 1] : tensor<9x25xf32> into tensor<18x25xf32>
// CHECK:           scf.yield %[[INS9]] : tensor<18x25xf32>
// CHECK:         }
// CHECK:         %[[INS:.+]] = tensor.insert_slice %[[R0]] into %[[OUT]][0, 0] [18, 25] [1, 1] : tensor<18x25xf32> into tensor<25x25xf32>
// CHECK:         %[[XS1:.+]] = tensor.extract_slice %[[IN1]][18, 0] [7, 34] [1, 1] : tensor<25x34xf32> to tensor<7x34xf32>
// CHECK:         %[[XS2:.+]] = tensor.extract_slice %[[INS]][18, 0] [7, 25] [1, 1] : tensor<25x25xf32> to tensor<7x25xf32>
// CHECK:         %[[XS3:.+]] = tensor.extract_slice %[[XS1]][0, 0] [4, 34] [1, 1] : tensor<7x34xf32> to tensor<4x34xf32>
// CHECK:         %[[XS4:.+]] = tensor.extract_slice %[[XS2]][0, 0] [4, 25] [1, 1] : tensor<7x25xf32> to tensor<4x25xf32>
// CHECK:         %[[R1:.+]] = linalg.matmul ins(%[[XS3]], %[[IN2]] : tensor<4x34xf32>, tensor<34x25xf32>) outs(%[[XS4]] : tensor<4x25xf32>) -> tensor<4x25xf32>
// CHECK:         %[[INS5:.+]] = tensor.insert_slice %[[R1]] into %[[XS2]][0, 0] [4, 25] [1, 1] : tensor<4x25xf32> into tensor<7x25xf32>
// CHECK:         %[[XS6:.+]] = tensor.extract_slice %[[XS1]][4, 0] [3, 34] [1, 1] : tensor<7x34xf32> to tensor<3x34xf32>
// CHECK:         %[[XS7:.+]] = tensor.extract_slice %[[INS5]][4, 0] [3, 25] [1, 1] : tensor<7x25xf32> to tensor<3x25xf32>
// CHECK:         %[[XS8:.+]] = tensor.extract_slice %[[XS6]][0, 0] [2, 34] [1, 1] : tensor<3x34xf32> to tensor<2x34xf32>
// CHECK:         %[[XS9:.+]] = tensor.extract_slice %[[XS7]][0, 0] [2, 25] [1, 1] : tensor<3x25xf32> to tensor<2x25xf32>
// CHECK:         %[[R2:.+]] = linalg.matmul ins(%[[XS8]], %[[IN2]] : tensor<2x34xf32>, tensor<34x25xf32>) outs(%[[XS9]] : tensor<2x25xf32>) -> tensor<2x25xf32>
// CHECK:         %[[INS10:.+]] = tensor.insert_slice %[[R2]] into %[[XS7]][0, 0] [2, 25] [1, 1] : tensor<2x25xf32> into tensor<3x25xf32>
// CHECK:         %[[XS11:.+]] = tensor.extract_slice %[[XS6]][2, 0] [1, 34] [1, 1] : tensor<3x34xf32> to tensor<1x34xf32>
// CHECK:         %[[XS12:.+]] = tensor.extract_slice %[[INS10]][2, 0] [1, 25] [1, 1] : tensor<3x25xf32> to tensor<1x25xf32>
// CHECK:         %[[R3:.+]] = linalg.matmul ins(%[[XS11]], %[[IN2]] : tensor<1x34xf32>, tensor<34x25xf32>) outs(%[[XS12]] : tensor<1x25xf32>) -> tensor<1x25xf32>
// CHECK:         %[[INS13:.+]] = tensor.insert_slice %[[R3]] into %[[INS10]][2, 0] [1, 25] [1, 1] : tensor<1x25xf32> into tensor<3x25xf32>
// CHECK:         %[[INS14:.+]] = tensor.insert_slice %[[INS13]] into %[[INS5]][4, 0] [3, 25] [1, 1] : tensor<3x25xf32> into tensor<7x25xf32>
// CHECK:         %[[INS15:.+]] = tensor.insert_slice %[[INS14]] into %[[INS]][18, 0] [7, 25] [1, 1] : tensor<7x25xf32> into tensor<25x25xf32>
// CHECK:         return %[[INS15]] : tensor<25x25xf32>

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tile_sizes, %chunk_sizes = transform.structured.continuous_tile_sizes %0 { dimension = 0, target_size = 9 } : (!transform.any_op) -> !transform.any_op
    %linalg_splits = transform.structured.split %0 after %chunk_sizes { dimension = 0, multiway } : !transform.any_op, !transform.any_op
    transform.foreach %linalg_splits, %tile_sizes with_zip_shortest : !transform.any_op, !transform.any_op {
    ^bb1(%linalg_split: !transform.any_op, %tile_size: !transform.any_op):
      %tiled_linalg_split, %dim0_loop = transform.structured.tile_using_for %linalg_split tile_sizes [%tile_size] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
    transform.yield
  }
}

func.func @continuous_tile_dynamic_linalg_matmul(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  return %0 : tensor<?x?xf32>
}

// CHECK:     #[[$MAP0:.*]] = affine_map<()[s0, s1] -> ((s0 floordiv 9) * 9, s1)>
// CHECK:     #[[$MAP3:.*]] = affine_map<()[s0, s1, s2] -> (((s0 mod 9) floordiv 8) * 8, s1 - s2)>
// CHECK:     #[[$MAP6:.*]] = affine_map<()[s0, s1, s2, s3] -> ((((s0 mod 9) mod 8) floordiv 4) * 4, s1 - s2 - s3)>
// CHECK:     #[[$MAP9:.*]] = affine_map<()[s0, s1, s2, s3, s4] -> ((((s0 mod 9) mod 4) floordiv 2) * 2, s1 - s2 - s3 - s4)>
// CHECK:     #[[$MAP12:.*]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> ((s0 mod 9) mod 2, s1 - s2 - s3 - s4 - s5)>
// CHECK-LABEL: @continuous_tile_dynamic_linalg_matmul
// CHECK-DAG: %[[C9:.*]] = arith.constant 9 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK:     %[[AM0:.*]] = affine.min #[[$MAP0]]()[%{{.*}}, %{{.*}}]
// CHECK:     %{{.*}} = scf.for %[[IDX:.+]] = %[[C0]] to %[[AM0]] step %[[C9]] iter_args(%[[OUT:.+]] = %{{.*}}) -> (tensor<?x?xf32>) {
// CHECK:       %[[MM:.+]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:       %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUT]][%[[IDX]], 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
// CHECK:     %[[AM4:.*]] = affine.min #[[$MAP3]]()[%{{.*}}, %{{.*}}, %[[AM0]]]
// CHECK:     %{{.*}} = scf.for %[[IDX:.+]] = %[[C0]] to %[[AM4]] step %[[C8]] iter_args(%[[OUT:.+]] = %{{.*}}) -> (tensor<?x?xf32>) {
// CHECK:       %[[MM:.+]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:       %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUT]][%[[IDX]], 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
// CHECK:     %[[AM8:.*]] = affine.min #[[$MAP6]]()[%{{.*}}, %{{.*}}, %[[AM0]], %[[AM4]]]
// CHECK:     %{{.*}} = scf.for %[[IDX:.+]] = %[[C0]] to %[[AM8]] step %[[C4]] iter_args(%[[OUT:.+]] = %{{.*}}) -> (tensor<?x?xf32>) {
// CHECK:       %[[MM:.+]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:       %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUT]][%[[IDX]], 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
// CHECK:     %[[AM12:.*]] = affine.min #[[$MAP9]]()[%{{.*}}, %{{.*}}, %[[AM0]], %[[AM4]], %[[AM8]]]
// CHECK:     %{{.*}} = scf.for %[[IDX:.+]] = %[[C0]] to %[[AM12]] step %[[C2]] iter_args(%[[OUT:.+]] = %{{.*}}) -> (tensor<?x?xf32>) {
// CHECK:       %[[MM:.+]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:       %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUT]][%[[IDX]], 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
// CHECK:     %[[AM16:.*]] = affine.min #[[$MAP12]]()[%{{.*}}, %{{.*}}, %[[AM0]], %[[AM4]], %[[AM8]], %[[AM12]]]
// CHECK:     %{{.*}} = scf.for %[[IDX:.+]] = %[[C0]] to %[[AM16]] step %[[C1]] iter_args(%[[OUT:.+]] = %{{.*}}) -> (tensor<?x?xf32>) {
// CHECK:       %[[MM:.+]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<1x?xf32>) -> tensor<1x?xf32>
// CHECK:       %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUT]][%[[IDX]], 0] [1, %{{.*}}] [1, 1] : tensor<1x?xf32> into tensor<?x?xf32>
