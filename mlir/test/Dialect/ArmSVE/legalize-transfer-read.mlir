// RUN: mlir-opt --arm-sve-legalize-vector-storage --split-input-file %s | FileCheck %s


// Test the `LegalizeTransferRead` pattern
// (mlir/lib/Dialect/ArmSVE/Transforms/LegalizeVectorStorage.cpp)

// -----

// This is the base case, unremarkable in any way, except that it's our main
// motivating example and use case.

// CHECK-LABEL:       @base_case
// CHECK-SAME:          %[[I:.+]]: index, %[[J:.+]]: index, %[[M:.+]]:
// CHECK-DAG:           %[[PAD:.+]] = arith.constant 123 : i8
// CHECK-DAG:           %[[C0:.+]] = arith.constant 0 : index
// CHECK:               %[[COLLAPSE:.+]] = memref.collapse_shape %[[M]]
// CHECK-SAME{LITERAL}:   [[0], [1], [2, 3]]
// CHECK-SAME:            : memref<?x?x?x8xi8> into memref<?x?x?xi8>
// CHECK-NEXT:          %[[T0:.+]] = vector.transfer_read %[[COLLAPSE]][%[[I]], %[[J]], %[[C0]]], %[[PAD]] {in_bounds = [true]}
// CHECK-SAME:            : memref<?x?x?xi8>, vector<[32]xi8>
// CHECK-NEXT:          %[[T1:.+]] = vector.shape_cast %[[T0]] : vector<[32]xi8> to vector<[4]x8xi8>
// CHECK-NEXT:          return %[[T1]] : vector<[4]x8xi8>

func.func @base_case(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x?x8xi8>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// Test the case where the scalable dimension is not the second-to-last.

// CHECK-LABEL:       @with_3d_vector
// CHECK-SAME:          %[[I:.+]]: index, %[[J:.+]]: index, %[[M:.+]]:
// CHECK-DAG:           %[[PAD:.+]] = arith.constant 123 : i8
// CHECK-DAG:           %[[COLLAPSED:.+]] = memref.collapse_shape %[[M]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2, 3]]
// CHECK-SAME:            : memref<?x?x2x8xi8> into memref<?x?xi8>
// CHECK-NEXT:          %[[T0:.+]] = vector.transfer_read %[[COLLAPSED]][%[[I]], %[[J]]], %[[PAD]] {in_bounds = [true]}
// CHECK-SAME:            : memref<?x?xi8>, vector<[64]xi8>
// CHECK-NEXT:          %[[T1:.+]] = vector.shape_cast %[[T0]] : vector<[64]xi8> to vector<[4]x2x8xi8>
// CHECK-NEXT:          return %[[T1]] : vector<[4]x2x8xi8>

func.func @with_3d_vector(%i : index, %j : index, %M : memref<?x?x2x8xi8>) -> vector<[4]x2x8xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true, true]} : memref<?x?x2x8xi8>, vector<[4]x2x8xi8>

  return %A : vector<[4]x2x8xi8>
}

// -----

// Test the case when the vector is already LLVM-legal (fixed).

// CHECK-LABEL: @negative_vector_legal_fixed
// CHECK-NOT: memref.collapse

func.func @negative_vector_legal_fixed(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<8x8xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x?x8xi8>, vector<8x8xi8>

  return %A : vector<8x8xi8>
}

// -----

// Test the case when the vector is already LLVM-legal (single-dimension scalable).

// CHECK-LABEL: @negative_vector_legal_1d_scalable
// CHECK-NOT: memref.collapse

func.func @negative_vector_legal_1d_scalable(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[8]xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true]} : memref<?x?x?x8xi8>, vector<[8]xi8>

  return %A : vector<[8]xi8>
}

// -----

// Test the case when the vector is already LLVM-legal (single trailing
// scalable dimension).

// CHECK-LABEL: @negative_vector_legal_trailing_scalable_dim
// CHECK-NOT: memref.collapse

func.func @negative_vector_legal_trailing_scalable_dim(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<8x[8]xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x?x8xi8>, vector<8x[8]xi8>

  return %A : vector<8x[8]xi8>
}

// -----

// Test the case of unsupported vector type (more than one scalable dimension)

// CHECK-LABEL: @negative_vector_type_two_scalable_dims
// CHECK-NOT: memref.collapse

func.func @negative_vector_type_two_scalable_dims(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[8]x[8]x8xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true, true]} : memref<?x?x?x8xi8>, vector<[8]x[8]x8xi8>

  return %A : vector<[8]x[8]x8xi8>
}

// -----

// Test the case of reading from a tensor - not supported, since the
// transform reasons about memory layouts.

// CHECK-LABEL: @negative_tensor_transfer
// CHECK-NOT: memref.collapse

func.func @negative_tensor_transfer(%i : index, %j : index, %M : tensor<?x?x?x8xi8>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true]} : tensor<?x?x?x8xi8>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// Test the case when the transfer is discontiguous because the memref
// is discontiguous.
// There are other ways to make a memref discontiguous. The transformation
// is not concerned with the particular reason a memref is discontiguous, but
// only with the fact. Therefore there are no variations with the memref made
// discontiguous by some other mechanism.

// CHECK-LABEL: @negative_discontig_mem
// CHECK-NOT: memref.collapse

#strides = strided<[?, ?, 16, 1]>

func.func @negative_discontig_mem(%i : index, %j : index, %M : memref<?x?x?x8xi8, #strides>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x?x8xi8, #strides>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// Test the case when the transformation is not applied because of
// a non-trivial permutation map (broadcast).

// CHECK-LABEL: @negative_broadcast
// CHECK-NOT: memref.collapse

#perm = affine_map<(i, j, k, p) -> (k, 0)>

func.func @negative_broadcast(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {permutation_map = #perm, in_bounds = [true, true] } : memref<?x?x?x8xi8>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// Test the case of a masked read - not supported right now.
// (see mlir/lib/Dialect/ArmSVE/Transforms/LegalizeVectorStorage.cpp)

// CHECK-LABEL: @negative_masked
// CHECK-NOT: memref.collapse

func.func @negative_masked(
  %i : index, %j : index,
  %M : memref<?x?x?x8xi8>, %mask : vector<[4]x8xi1>) -> vector<[4]x8xi8> {
  
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.mask %mask {
    vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true] } : memref<?x?x?x8xi8>, vector<[4]x8xi8>
  } : vector<[4]x8xi1> -> vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// Test case with a mask operand - not supported right now.
// (see mlir/lib/Dialect/ArmSVE/Transforms/LegalizeVectorStorage.cpp)

// CHECK-LABEL: @negative_with_mask
// CHECK-NOT: memref.collapse

func.func @negative_with_mask(
  %i : index, %j : index,
  %M : memref<?x?x?x8xi8>, %mask : vector<[4]x8xi1>) -> vector<[4]x8xi8> {
  
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad, %mask {in_bounds = [true, true] } : memref<?x?x?x8xi8>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// Test the case when the dimensions to collapse (excluding the scalable one)
// of the vector and the memref do not match (static non matching dimension).

// CHECK-LABEL: @negative_non_matching_dim_static
// CHECK-NOT: memref.collapse

func.func @negative_non_matching_dim_static(%i : index, %j : index,  %M : memref<?x?x?x8xi8>) -> vector<[4]x4xi8> {
  
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true] } : memref<?x?x?x8xi8>, vector<[4]x4xi8>

  return %A : vector<[4]x4xi8>
}

// -----

// Test the case when the dimensions to collapse (excluding the scalable one)
// of the vector and the memref do not match (dynamic non matching dimension).

// CHECK-LABEL: @negative_non_matching_dim_dynamic
// CHECK-NOT: memref.collapse

func.func @negative_non_matching_dim_dynamic(%i : index, %j : index,  %M : memref<?x?x?x?xi8>) -> vector<[4]x4xi8> {
  
  %c0 = arith.constant 0 : index
  %pad = arith.constant 123 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %pad {in_bounds = [true, true] } : memref<?x?x?x?xi8>, vector<[4]x4xi8>

  return %A : vector<[4]x4xi8>
}
