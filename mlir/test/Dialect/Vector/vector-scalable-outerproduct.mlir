// RUN: mlir-opt -split-input-file -verify-diagnostics %s | mlir-opt

func.func @scalable_outerproduct(%src : memref<?xf32>) {
  %idx = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %0 = vector.load %src[%idx] : memref<?xf32>, vector<[4]xf32>
  %1 = vector.load %src[%idx] : memref<?xf32>, vector<[4]xf32>

  %op = vector.outerproduct %0, %1 : vector<[4]xf32>, vector<[4]xf32>
  vector.store %op, %src[%idx] : memref<?xf32>, vector<[4]x[4]xf32>

  %op2 = vector.outerproduct %0, %cst : vector<[4]xf32>, f32
  vector.store %op2, %src[%idx] : memref<?xf32>, vector<[4]xf32>
  return
}

// -----

func.func @invalid_outerproduct(%src : memref<?xf32>) {
  %idx = arith.constant 0 : index
  %0 = vector.load %src[%idx] : memref<?xf32>, vector<[4]xf32>
  %1 = vector.load %src[%idx] : memref<?xf32>, vector<4xf32>

  // expected-error @+1 {{expected either both or only #2 operand dim to be scalable}}
  %op = vector.outerproduct %0, %1 : vector<[4]xf32>, vector<4xf32>

  return
}

// -----

func.func @invalid_outerproduct1(%src : memref<?xf32>) {
  %idx = arith.constant 0 : index
  %0 = vector.load %src[%idx] : memref<?xf32>, vector<[4]x[4]xf32>
  %1 = vector.load %src[%idx] : memref<?xf32>, vector<[4]xf32>

  // expected-error @+1 {{'vector.outerproduct' op expected 1-d vector for operand #1}}
  %op = vector.outerproduct %0, %1 : vector<[4]x[4]xf32>, vector<[4]xf32>
}
