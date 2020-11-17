// RUN: mlir-opt %s -convert-std-to-llvm -verify-diagnostics -split-input-file

func @mlir_cast_to_llvm(%0 : index) -> !llvm.i64 {
  // expected-error@+1 {{'llvm.mlir.cast' op type must be non-index integer types, float types, or vector of mentioned types}}
  %1 = llvm.mlir.cast %0 : index to !llvm.i64
  return %1 : !llvm.i64
}

// -----

func @mlir_cast_from_llvm(%0 : !llvm.i64) -> index {
  // expected-error@+1 {{'llvm.mlir.cast' op type must be non-index integer types, float types, or vector of mentioned types}}
  %1 = llvm.mlir.cast %0 : !llvm.i64 to index
  return %1 : index
}

// -----

func @mlir_cast_to_llvm_int(%0 : i32) -> !llvm.i64 {
  // expected-error@+1 {{failed to legalize operation 'llvm.mlir.cast' that was explicitly marked illegal}}
  %1 = llvm.mlir.cast %0 : i32 to !llvm.i64
  return %1 : !llvm.i64
}

// -----

func @mlir_cast_to_llvm_vec(%0 : vector<1x1xf32>) -> !llvm.vec<1 x float> {
  // expected-error@+1 {{'llvm.mlir.cast' op only 1-d vector is allowed}}
  %1 = llvm.mlir.cast %0 : vector<1x1xf32> to !llvm.vec<1 x float>
  return %1 : !llvm.vec<1 x float>
}

// -----

// Should not crash on unsupported types in function signatures.
func private @unsupported_signature() -> tensor<10 x i32>

// -----

func private @partially_supported_signature() -> (vector<10 x i32>, tensor<10 x i32>)
