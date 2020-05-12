// RUN: mlir-opt %s -convert-std-to-llvm -verify-diagnostics -split-input-file

#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

func @invalid_memref_cast(%arg0: memref<?x?xf64>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  // expected-error@+1 {{'std.memref_cast' op operand #0 must be unranked.memref of any type values or memref of any type values, but got '!llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">'}}
  %5 = memref_cast %arg0 : memref<?x?xf64> to memref<?x?xf64, #map1>
  %25 = std.subview %5[%c0, %c0][%c1, %c1][1, 1] : memref<?x?xf64, #map1> to memref<?x?xf64, #map1>
  return
}

// -----

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

func @mlir_cast_to_llvm_vec(%0 : vector<1x1xf32>) -> !llvm<"<1 x float>"> {
  // expected-error@+1 {{'llvm.mlir.cast' op only 1-d vector is allowed}}
  %1 = llvm.mlir.cast %0 : vector<1x1xf32> to !llvm<"<1 x float>">
  return %1 : !llvm<"<1 x float>">
}
