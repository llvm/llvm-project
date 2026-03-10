// RUN: mlir-opt %s --inline --sccp --canonicalize -split-input-file -verify-diagnostics

func.func @get_tile_size() -> index {
  %c0 = arith.constant 0 : index
  return %c0 : index
}

func.func private @use(%A: tensor<?x16x?x1xi32>)

func.func @pack(%A: tensor<7x16xi32>) {
  %c1 = arith.constant 1 : index
  %pad_val = arith.constant 123 : i32
  %tile_size = func.call @get_tile_size() : () -> index
  %empty = tensor.empty(%c1, %tile_size) : tensor<?x16x?x1xi32>
  // expected-error @below {{invalid zero tile factor}}
  %pack = linalg.pack %A
    padding_value(%pad_val : i32)
    inner_dims_pos = [0, 1]
    inner_tiles = [%tile_size, 1]
    into %empty : tensor<7x16xi32> -> tensor<?x16x?x1xi32>
  func.call @use(%pack) : (tensor<?x16x?x1xi32>) -> ()
  return
}