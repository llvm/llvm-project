// RUN: mlir-opt <%s -split-input-file -verify-diagnostics -canonicalize

// -----

func.func @indirectly_generate_negative_size() -> tensor<?x8xi32> {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %size = affine.max affine_map<(d0) -> (d0 mod 64 - 8)>(%c0)
  // expected-error@+1 {{tensor dimensions must be non-negative}}
  %tensor = tensor.generate %size {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : i32
  } : tensor<?x8xi32>
  return %tensor : tensor<?x8xi32>
}
