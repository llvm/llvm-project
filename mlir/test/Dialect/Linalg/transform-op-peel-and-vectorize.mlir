// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize | FileCheck %s

// Demonstrates what happens when peeling the middle loop (2nd parallel
// dimension) followed by vectorization in the presence of _scalable_ vectors
// (these are introduced through scalable tiling). The main goal is to verify
// that canonicalizations fold away the masks in the main loop.

func.func @matmul(%A: tensor<1024x512xf32>,
                  %B: tensor<512x2000xf32>,
                  %C: tensor<1024x2000xf32>) -> tensor<1024x2000xf32> {

// CHECK:      #[[MAP:.*]] = affine_map<()[s0] -> (-(2000 mod s0) + 2000)>
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[C2000:.*]] = arith.constant 2000 : index
// CHECK-DAG:  %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:  %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG:  %[[C512:.*]] = arith.constant 512 : index
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C16:.*]] = arith.constant 16 : index
// CHECK:      %[[VSCALE:.*]] = vector.vscale
// CHECK:      %[[STEP:.*]] = arith.muli %[[VSCALE]], %[[C16]] : index
// CHECK:      scf.for {{.*}} %[[C0]] to %[[C1024]] step %[[C8]] iter_args(%arg4 = %arg2) -> (tensor<1024x2000xf32>) {

// Main loop after vectorisation (without masking)

// CHECK:         %[[UB_MAIN:.*]] = affine.apply #[[MAP]]()[%[[STEP]]]
// CHECK:         scf.for {{.*}} %[[C0]] to %[[UB_MAIN]] step %[[STEP]] {{.*}} -> (tensor<1024x2000xf32>) {
// CHECK:           scf.for %arg7 = %[[C0]] to %[[C512]] step %[[C1]] {{.*}} -> (tensor<1024x2000xf32>) {
// CHECK-NOT:         vector.mask
// CHECK:             arith.mulf {{.*}} : vector<8x[16]x1xf32>
// CHECK-NEXT:        vector.shape_cast {{.*}} : vector<8x[16]x1xf32> to vector<8x[16]xf32>
// CHECK-NEXT:        arith.addf {{.*}} : vector<8x[16]xf32>
// CHECK-NOT:         vector.mask
// CHECK:             scf.yield {{.*}} : tensor<1024x2000xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield {{.*}} : tensor<1024x2000xf32>
// CHECK-NEXT:    }

// Remainder loop after vectorisation (with masking)

// CHECK:       scf.for {{.*}} %[[UB_MAIN]] to %[[C2000]] step %[[STEP]] {{.*}} -> (tensor<1024x2000xf32>) {
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C512]] step %[[C1]] {{.*}} -> (tensor<1024x2000xf32>) {
// CHECK:           %[[MASK_1:.*]] = vector.create_mask {{.*}} : vector<1x[16]xi1>
// CHECK:           %[[RHS:.*]] = vector.mask %[[MASK_1]] { vector.transfer_read {{.*}} } : vector<1x[16]xi1> -> vector<8x[16]x1xf32>
// CHECK:           %[[MASK_2:.*]] = vector.create_mask {{.*}} : vector<8x[16]xi1>
// CHECK:           %[[LHS:.*]] = vector.mask %[[MASK_2]] { vector.transfer_read {{.*}} } : vector<8x[16]xi1> -> vector<8x[16]xf32>
// CHECK:           %[[MUL:.*]] = arith.mulf %{{.*}}, %[[RHS]] : vector<8x[16]x1xf32>
// CHECK:           %[[MASK_3:.*]] = vector.create_mask {{.*}} : vector<8x[16]xi1>
// CHECK:           vector.shape_cast %[[MUL]] : vector<8x[16]x1xf32> to vector<8x[16]xf32>
// CHECK:           arith.addf %[[LHS]], %{{.*}} : vector<8x[16]xf32>
// CHECK:           arith.select %[[MASK_3]], {{.*}} : vector<8x[16]xi1>, vector<8x[16]xf32>
// CHECK:           vector.mask %[[MASK_2]] { vector.transfer_write {{.*}} } : vector<8x[16]xi1> -> tensor<8x?xf32>
// CHECK:           scf.yield %inserted_slice : tensor<1024x2000xf32>
// CHECK:         }
// CHECK:         scf.yield {{.*}} : tensor<1024x2000xf32>
// CHECK:       }
// CHECK:       scf.yield {{.*}} : tensor<1024x2000xf32>
// CHECK-NEXT:    }

  %res = linalg.matmul ins(%A, %B: tensor<1024x512xf32>, tensor<512x2000xf32>)
            outs(%C: tensor<1024x2000xf32>) -> tensor<1024x2000xf32>
  return %res : tensor<1024x2000xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    // 1. Scalable tiling
    %_, %loop_1, %loop_2, %loop_3 =
      transform.structured.tile_using_for %matmul tile_sizes [8, [16], 1] : (!transform.any_op)
      -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">,!transform.op<"scf.for">)

    // 2. Loop peeling (only the middle dimension)
    %main_loop, %remainder_loop = transform.loop.peel %loop_2 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)

    // 3. Vectorize the main loop
    %matmul_main = transform.structured.match ops{["linalg.matmul"]} in %main_loop : (!transform.op<"scf.for">) -> !transform.any_op
    transform.structured.vectorize %matmul_main vector_sizes [8, [16], 1] : !transform.any_op

    // 4. Vectorize the remainder loop
    %matmul_remainder = transform.structured.match ops{["linalg.matmul"]} in %remainder_loop : (!transform.op<"scf.for">) -> !transform.any_op
    transform.structured.vectorize %matmul_remainder vector_sizes [8, [16], 1] : !transform.any_op

    transform.yield
  }
}
