// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:   -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline -cse -canonicalize -convert-vector-to-scf \
// DEFINE:   -convert-vector-to-llvm="enable-arm-neon enable-arm-i8mm" -test-lower-to-llvm \
// DEFINE: -o %t

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e main -entry-point-result=void --march=aarch64 --mattr="+neon,+i8mm" \
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

//===----------------------------------------------------------------------===//
// Tiles, vectorizes and lowers a `linalg.matmul` down to Arm's FEAT_I8MM
// `smmla` instruction via `transform.apply_patterns.arm_neon.vector_contract_to_i8mm`
// (LowerContractionToNeonI8MMPattern).
//
// That pattern expects a `vector.contract` with LHS vector<MxKxi8>, RHS
// vector<NxKxi8> (RHS read "N-major", i.e. logically transposed relative to
// what a plain `linalg.matmul` produces), and ACC/OUT vector<MxNxi32> -- see
// #packed_maps in Vector/CPU/ArmNeon/vector-contract-i8mm.mlir. To get there,
// this test gives `linalg.matmul` an explicit `indexing_maps` attribute that
// reads the second operand "N-major" (== `MatmulTransposeBOp`'s default
// maps, see LinalgOps.cpp), feeding it the second input pre-transposed
// (NxK instead of KxN). This mirrors how ArmSME/matmul-transpose-a.mlir
// transposes the *LHS* instead, for SME's own hardware constraints.
//===----------------------------------------------------------------------===//

// CHECK-IR-LABEL: llvm.func @main
// CHECK-IR-COUNT-4: arm_neon.intr.smmla
func.func @main() {
  // A: MxK = 4x8.
  %A = arith.constant dense<[
    [-35, -27, -36, -31,  23, -34,  -8, -33],
    [-20,  17, -32, -47,  37,  22,  -7, -21],
    [ -7, -35,  20,  -4,  39,  46, -23,  40],
    [ 40,  27,  37,  43,  38,  -6,  37,  49]
  ]> : tensor<4x8xi8>

  // B, transposed: NxK = 4x8 (row n holds column n of the logical KxN RHS).
  %Bt = arith.constant dense<[
    [-17, -50,  -1,  48, -13,  22,  39,  33],
    [-35, -24,  37, -32,  33,  30, -11, -17],
    [-28,  31,   3, -44, -15, -27,  22,  35],
    [-23,  39,  48,  26, -23,  32, -39, -38]
  ]> : tensor<4x8xi8>

  // C: MxN = 4x4, non-zero to also exercise the "+ ACC" part of `smmla`.
  %C = arith.constant dense<[
    [-44,  20,  44, -46],
    [ -8,  25, -34,  26],
    [-20, -36,  -3,  39],
    [-48, -31, -25, -21]
  ]> : tensor<4x4xi32>

  %A_dyn = tensor.cast %A : tensor<4x8xi8> to tensor<?x?xi8>
  %Bt_dyn = tensor.cast %Bt : tensor<4x8xi8> to tensor<?x?xi8>
  %C_dyn = tensor.cast %C : tensor<4x4xi32> to tensor<?x?xi32>

  %res = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%A_dyn, %Bt_dyn : tensor<?x?xi8>, tensor<?x?xi8>)
      outs(%C_dyn : tensor<?x?xi32>) -> tensor<?x?xi32>

  // Print and verify the output
  // CHECK-LABEL: NEON: START OF TEST OUTPUT
  vector.print str "NEON: START OF TEST OUTPUT\n"

  // CHECK-NEXT: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [-1999,    1941,     685,   -2879]
  // CHECK-NEXT: [-3705,    2952,     987,    -685]
  // CHECK-NEXT: [2565,     4157,   -1589,    -357]
  // CHECK-NEXT: [2383,    -2252,      32,   -1365]
  %xf = tensor.cast %res : tensor<?x?xi32> to tensor<*xi32>
  call @printMemrefI32(%xf) : (tensor<*xi32>) -> ()

  // CHECK-NEXT: NEON: END OF TEST OUTPUT
  vector.print str "NEON: END OF TEST OUTPUT\n"

  return
}

module attributes {transform.with_named_sequence} {
  // Tile, vectorize, then lower the `vector.contract` straight to FEAT_I8MM ops.
  transform.named_sequence @tile_and_vectorize_matmul(%func
    : !transform.op<"func.func"> {transform.readonly}) {

    // Step 0: Get a handle to the matmul op, if any.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.op<"func.func">) -> !transform.any_op

    // Step 1: Tile to the FEAT_I8MM tile shape (M=N=4, K=8). This is the
    // whole problem size here, so tiling produces a single tile, no tail.
    %tiled_matmul, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [4, 4, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 2: Vectorize directly to a named `vector.contract`.
    transform.structured.vectorize %tiled_matmul vector_sizes [4, 4, 8]
      {create_named_contraction} : !transform.any_op

    // Step 3: M, N, K are static and match the tile/vector sizes, so
    // vectorization masks are trivially full tile; clean them up.
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.sink_ops
    } : !transform.op<"func.func">

    // Step 4: Lower `vector.contract` straight to FEAT_I8MM ops, 
    // instead of the generic outerproduct lowering, which would 
    // exercise `smmla`.
    transform.apply_patterns to %func {
      transform.apply_patterns.arm_neon.vector_contract_to_i8mm
    } : !transform.op<"func.func">

    transform.yield
  }

  // Apply `tile_and_vectorize_matmul` to every function in the module.
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.op<"func.func">

    transform.foreach %funcs : !transform.op<"func.func"> {
      ^bb0(%func : !transform.op<"func.func">):
        transform.include @tile_and_vectorize_matmul failures(propagate)
          (%func) : (!transform.op<"func.func">) -> ()
    }
    transform.yield
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
