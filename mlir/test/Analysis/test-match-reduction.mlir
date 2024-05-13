// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-match-reduction))" -verify-diagnostics -split-input-file

// Verify that the generic reduction detection utility works on different
// dialects.

// expected-remark@below {{Testing function}}
func.func @linalg_red_add(%in0t : tensor<?xf32>, %out0t : tensor<1xf32>) {
  // expected-remark@below {{Reduction found in output #0!}}
  // expected-remark@below {{Reduced Value: <block argument> of type 'f32' at index: 0}}
  // expected-remark@below {{Combiner Op: %1 = arith.addf }}
  %red = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                          affine_map<(d0) -> (0)>],
                                          iterator_types = ["reduction"]}
   ins(%in0t : tensor<?xf32>)
   outs(%out0t : tensor<1xf32>) {
    ^bb0(%in0: f32, %out0: f32):
      %add = arith.addf %in0, %out0 : f32
      linalg.yield %add : f32
    } -> tensor<1xf32>
  return
}

// -----

// expected-remark@below {{Testing function}}
func.func @affine_red_add(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   // expected-remark@below {{Reduction found in output #0!}}
   // expected-remark@below {{Reduced Value: %1 = affine.load }}
   // expected-remark@below {{Combiner Op: %2 = arith.addf }}
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// -----

// TODO: Iteration-carried values with multiple uses are not supported yet.
// expected-remark@below {{Testing function}}
func.func @linalg_red_max(%in0t: tensor<4x4xf32>, %out0t: tensor<4xf32>) {
  // expected-remark@below {{Reduction NOT found in output #0!}}
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%in0t : tensor<4x4xf32>)
   outs(%out0t : tensor<4xf32>) {
    ^bb0(%in0: f32, %out0: f32):
      %cmp = arith.cmpf ogt, %in0, %out0 : f32
      %sel = arith.select %cmp, %in0, %out0 : f32
      linalg.yield %sel : f32
    } -> tensor<4xf32>
  return
}

// -----

// expected-remark@below {{Testing function}}
func.func @linalg_fused_red_add(%in0t: tensor<4x4xf32>, %out0t: tensor<4xf32>) {
  // expected-remark@below {{Reduction found in output #0!}}
  // expected-remark@below {{Reduced Value: %2 = arith.subf}}
  // expected-remark@below {{Combiner Op: %3 = arith.addf}}
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%in0t : tensor<4x4xf32>)
   outs(%out0t : tensor<4xf32>) {
    ^bb0(%in0: f32, %out0: f32):
      %mul = arith.mulf %in0, %in0 : f32
      %sub = arith.subf %mul, %in0 : f32
      %add = arith.addf %sub, %out0 : f32
      linalg.yield %add : f32
    } -> tensor<4xf32>
  return
}

// -----

// expected-remark@below {{Testing function}}
func.func @affine_no_red_rec(%in: memref<512xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 // %rec is the value loaded in the previous iteration.
 // expected-remark@below {{Reduction NOT found in output #0!}}
 %final_val = affine.for %j = 0 to 512 iter_args(%rec = %cst) -> (f32) {
   %ld = affine.load %in[%j] : memref<512xf32>
   %add = arith.addf %ld, %rec : f32
   affine.yield %ld : f32
 }
 return
}

// -----

// expected-remark@below {{Testing function}}
func.func @affine_output_dep(%in: memref<512xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 // Reduction %red is not supported because it depends on another
 // loop-carried dependence.
 // expected-remark@below {{Reduction NOT found in output #0!}}
 // expected-remark@below {{Reduction NOT found in output #1!}}
 %final_red, %final_dep = affine.for %j = 0 to 512
  iter_args(%red = %cst, %dep = %cst) -> (f32, f32) {
   %ld = affine.load %in[%j] : memref<512xf32>
   %add = arith.addf %dep, %red : f32
   affine.yield %add, %ld : f32, f32
 }
 return
}

