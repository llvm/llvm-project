// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa --verify-diagnostics %s

func.func @graph_constant_id_i64() -> tensor<17xi32> {
  // expected-error@below {{'tosa.const' op requires `grapharm.graph_constant_id` to be a signless i32 integer attribute}}
  %res = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<17xi32>}> {grapharm.graph_constant_id = 7 : i64} : () -> tensor<17xi32>
  return %res : tensor<17xi32>
}

// -----

func.func @graph_constant_id_si32() -> tensor<17xi32> {
  // expected-error@below {{'tosa.const' op requires `grapharm.graph_constant_id` to be a signless i32 integer attribute}}
  %res = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<17xi32>}> {grapharm.graph_constant_id = 7 : si32} : () -> tensor<17xi32>
  return %res : tensor<17xi32>
}

// -----

func.func @graph_constant_id_ui32() -> tensor<17xi32> {
  // expected-error@below {{'tosa.const' op requires `grapharm.graph_constant_id` to be a signless i32 integer attribute}}
  %res = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<17xi32>}> {grapharm.graph_constant_id = 7 : ui32} : () -> tensor<17xi32>
  return %res : tensor<17xi32>
}
