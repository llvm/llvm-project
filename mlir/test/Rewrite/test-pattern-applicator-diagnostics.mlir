// RUN: mlir-opt %s -canonicalize -mlir-emit-pattern-match-diagnostics="match-success" -verify-diagnostics

func.func @tensor_empty_canonicalization() -> tensor<?xf32> {
  %c4 = arith.constant 4 : index
  // Do not match "(anonymous namespace)::", as it may render differently
  // depending on the C++ compiler.
  // expected-remark-re @+1 {{pattern match success: {{.*}}ReplaceEmptyTensorStaticShapeDims}}
  %r = tensor.empty(%c4) : tensor<?xf32>
  return %r : tensor<?xf32>
}
