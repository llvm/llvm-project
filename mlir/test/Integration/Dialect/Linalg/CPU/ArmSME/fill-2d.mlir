// RUN: mlir-opt %s \
// RUN:   -transform-interpreter \
// RUN:   -test-transform-dialect-erase-schedule \
// RUN:   -lower-vector-mask \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -enable-arm-streaming="streaming-mode=streaming-locally za-mode=new-za" \
// RUN:   -convert-vector-to-arm-sme -allocate-arm-sme-tiles -convert-arm-sme-to-scf \
// RUN:   -convert-arm-sme-to-llvm -cse -canonicalize \
// RUN:   -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd \
// RUN:   -e=entry -entry-point-result=void \
// RUN:   -march=aarch64 -mattr="+sve,+sme" \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib | \
// RUN: FileCheck %s

func.func @entry() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %step = arith.constant 1 : index

  %c123_f32 = arith.constant 123.0 : f32

  %min_elts_s = arith.constant 4 : index
  %vscale = vector.vscale

  // "svl" refers to the Streaming Vector Length and "svl_s" the number of
  // 32-bit elements in a vector of SVL bits.
  %svl_s = arith.muli %min_elts_s, %vscale : index

  %tile_init = bufferization.alloc_tensor(%svl_s, %svl_s) : tensor<?x?xf32>

  // Initialize tile with "123.0".
  // TODO: this could be simplified to tensor.splat + tensor.insert_slice once
  // splat supports dynamically shaped tensors.
  %tile_0 = scf.for %i = %c0 to %svl_s step %step iter_args(%tile_partial = %tile_init) -> tensor<?x?xf32> {
    %inner_tile = scf.for %j = %c0 to %svl_s step %step iter_args(%inner_tile_partial = %tile_partial) -> tensor<?x?xf32> {
      %tile_update = tensor.insert %c123_f32 into %inner_tile_partial[%i, %j] : tensor<?x?xf32>
      scf.yield %tile_update : tensor<?x?xf32>
    }
    scf.yield %inner_tile : tensor<?x?xf32>
  }

  // Print tile after initialization. The smallest SVL is 128-bits so the tile
  // will be at least 4x4xf32.
  //
  // CHECK:      ( 123, 123, 123, 123
  // CHECK-NEXT: ( 123, 123, 123, 123
  // CHECK-NEXT: ( 123, 123, 123, 123
  // CHECK-NEXT: ( 123, 123, 123, 123
  scf.for %i = %c0 to %svl_s step %step {
    vector.print punctuation <open>
    scf.for %j = %c0 to %svl_s step %step {
      %element = tensor.extract %tile_0[%i, %j] : tensor<?x?xf32>
      vector.print %element : f32 punctuation <no_punctuation>

      // Print comma unless last element.
      %c1_index = arith.constant 1 : index
      %last_i = arith.subi %svl_s, %c1_index : index
      %isNotLastIter = arith.cmpi ult, %j, %last_i : index
      scf.if %isNotLastIter {
        vector.print punctuation <comma>
      }
    }
    vector.print punctuation <close>
    vector.print punctuation <newline>
  }

  // Fill tile with pi.
  %pi = arith.constant 3.14 : f32
  %tile_1 = linalg.fill ins(%pi : f32) outs(%tile_0 : tensor<?x?xf32>) -> tensor<?x?xf32>

  // Print tile after filling with pi. The smallest SVL is 128-bits so the tile
  // will be at least 4x4xf32.
  //
  // CHECK:      ( 3.14, 3.14, 3.14, 3.14
  // CHECK-NEXT: ( 3.14, 3.14, 3.14, 3.14
  // CHECK-NEXT: ( 3.14, 3.14, 3.14, 3.14
  // CHECK-NEXT: ( 3.14, 3.14, 3.14, 3.14
  scf.for %i = %c0 to %svl_s step %step {
    vector.print punctuation <open>
    scf.for %j = %c0 to %svl_s step %step {
      %element = tensor.extract %tile_1[%i, %j] : tensor<?x?xf32>
      vector.print %element : f32 punctuation <no_punctuation>

      // Print comma unless last element.
      %c1_index = arith.constant 1 : index
      %last_i = arith.subi %svl_s, %c1_index : index
      %isNotLastIter = arith.cmpi ult, %j, %last_i : index
      scf.if %isNotLastIter {
        vector.print punctuation <comma>
      }
    }
    vector.print punctuation <close>
    vector.print punctuation <newline>
  }

  // CHECK: SME: END OF TEST OUTPUT
  vector.print str "SME: END OF TEST OUTPUT"

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [[4], [4]] : !transform.any_op
    transform.yield
  }
}
