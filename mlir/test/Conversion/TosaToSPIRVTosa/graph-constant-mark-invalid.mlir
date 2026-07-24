// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa-mark-graph-constants --verify-diagnostics %s

func.func @pre_marked_const() -> tensor<17xi32> {
  // expected-error@below {{'tosa.const' op already has `grapharm.graph_constant_id`; this pass assigns graph constant IDs automatically and does not support pre-marked constants}}
  %res = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<17xi32>}> {grapharm.graph_constant_id = 7 : i32} : () -> tensor<17xi32>
  return %res : tensor<17xi32>
}

// -----

func.func @pre_marked_const_shape() -> !tosa.shape<33> {
  // expected-error@below {{'tosa.const_shape' op already has `grapharm.graph_constant_id`; this pass assigns graph constant IDs automatically and does not support pre-marked constants}}
  %res = "tosa.const_shape"() <{values = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]> : tensor<33xindex>}> {grapharm.graph_constant_id = 8 : i32} : () -> !tosa.shape<33>
  return %res : !tosa.shape<33>
}
