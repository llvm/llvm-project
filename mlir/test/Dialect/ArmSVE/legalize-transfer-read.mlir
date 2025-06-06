// RUN: mlir-opt --arm-sve-legalize-vector-storage --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL:       @test_base_case
// CHECK-SAME:          %[[I:arg0]]: index, %[[J:arg1]]: index, %[[M:arg2]]:
// CHECK:               %[[COLLAPSE:.+]] = memref.collapse_shape %[[M]]
// CHECK-SAME{LITERAL}:   [[0], [1], [2, 3]]
// CHECK-SAME:            : memref<?x?x?x8xi8> into memref<?x?x?xi8>
// CHECK-NEXT:          %[[T0:.+]] = vector.transfer_read %[[COLLAPSE]][%[[I]], %[[J]], %c0], %c0_i8 {in_bounds = [true]}
// CHECK-SAME:            : memref<?x?x?xi8>, vector<[32]xi8>
// CHECK-NEXT:          %[[T1:.+]] = vector.shape_cast %[[T0]] : vector<[32]xi8> to vector<[4]x8xi8>
// CHECK-NEXT:          return %[[T1]] : vector<[4]x8xi8>

func.func @test_base_case(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?x8xi8>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// CHECK-LABEL:       @test_using_strided_layout
// CHECK-SAME:          %[[I:arg0]]: index, %[[J:arg1]]: index, %[[M:arg2]]
// CHECK:               %[[COLLAPSE:.+]] = memref.collapse_shape %[[M]]
// CHECK-SAME{LITERAL}:   [[0], [1], [2, 3]]
// CHECK-SAME:            : memref<?x?x?x8xi8, strided<[?, ?, 8, 1]>> into
// CHECK-SAME:              memref<?x?x?xi8, strided<[?, ?, 1]>>
// CHECK-NEXT:          %[[T0:.+]] = vector.transfer_read %[[COLLAPSE]][%[[I]], %[[J]], %c0], %c0_i8 {in_bounds = [true]}
// CHECK-SAME:            : memref<?x?x?xi8, strided<[?, ?, 1]>>, vector<[32]xi8>
// CHECK-NEXT:          %[[T1:.+]] = vector.shape_cast %[[T0]] : vector<[32]xi8> to vector<[4]x8xi8>
// CHECK-NEXT:          return %[[T1]] : vector<[4]x8xi8>

#s0 = strided<[?, ?, 8, 1]>

func.func @test_using_strided_layout(%i : index, %j : index, %M : memref<?x?x?x8xi8, #s0>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?x8xi8, #s0>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// CHECK-LABEL:       @test_3d_vector
// CHECK-SAME:          %[[I:arg0]]: index, %[[J:arg1]]: index, %[[M:arg2]]
// CHECK:               %[[COLLAPSED:.+]] = memref.collapse_shape %[[M]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2, 3]]
// CHECK-SAME:            : memref<?x?x2x8xi8, strided<[?, 16, 8, 1]>> into
// CHECK-SAME:              memref<?x?xi8, strided<[?, 1]>>
// CHECK-NEXT:          %[[T0:.+]] = vector.transfer_read %[[COLLAPSED]][%[[I]], %[[J]]], %c0_i8 {in_bounds = [true]}
// CHECK-SAME:            : memref<?x?xi8, strided<[?, 1]>>, vector<[64]xi8>
// CHECK-NEXT:          %[[T1:.+]] = vector.shape_cast %[[T0]] : vector<[64]xi8> to vector<[4]x2x8xi8>
// CHECK-NEXT:          return %[[T1]] : vector<[4]x2x8xi8>

#s1 = strided<[?, 16, 8, 1]>

func.func @test_3d_vector(%i : index, %j : index, %M : memref<?x?x2x8xi8, #s1>) -> vector<[4]x2x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<?x?x2x8xi8, #s1>, vector<[4]x2x8xi8>

  return %A : vector<[4]x2x8xi8>
}

// -----

// CHECK-LABEL:       @test_4d_vector
// CHECK-SAME:          %[[I:arg0]]: index, %[[J:arg1]]: index, %[[M:arg2]]
// CHECK:               %[[COLLAPSED:.+]] = memref.collapse_shape %[[M]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2, 3]]
// CHECK-SAME:           : memref<?x?x2x8xi8, strided<[?, 16, 8, 1]>> into
// CHECK-SAME:             memref<?x?xi8, strided<[?, 1]>>
// CHECK-NEXT:         %[[T0:.+]] = vector.transfer_read %[[COLLAPSED]][%[[I]], %[[J]]], %c0_i8 {in_bounds = [false, true]}
// CHECK-SAME:           : memref<?x?xi8, strided<[?, 1]>>, vector<2x[64]xi8>
// CHECK-NEXT:         %[[T1:.+]] = vector.shape_cast %[[T0]] : vector<2x[64]xi8> to vector<2x[4]x2x8xi8>
// CHECK-NEXT:         return %[[T1]] : vector<2x[4]x2x8xi8>

#s2 = strided<[?, 16, 8, 1]>

func.func @test_4d_vector(%i : index, %j : index, %M : memref<?x?x2x8xi8, #s2>) -> vector<2x[4]x2x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [false, true, true, true]} : memref<?x?x2x8xi8, #s2>, vector<2x[4]x2x8xi8>

  return %A : vector<2x[4]x2x8xi8>
}

// -----

// CHECK-LABEL: @negative_test_vector_legal_non_scalable
// CHECK-NOT: memref.collapse

func.func @negative_test_vector_legal_non_scalable(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<8x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?x8xi8>, vector<8x8xi8>

  return %A : vector<8x8xi8>
}

// -----

// CHECK-LABEL: @negative_test_vector_legal_scalable_0
// CHECK-NOT: memref.collapse

func.func @negative_test_vector_legal_scalable_0(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[8]xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true]} : memref<?x?x?x8xi8>, vector<[8]xi8>

  return %A : vector<[8]xi8>
}

// -----

// CHECK-LABEL: @negative_test_vector_legal_scalable_1
// CHECK-NOT: memref.collapse

func.func @negative_test_vector_legal_scalable_1(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<8x[8]xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?x8xi8>, vector<8x[8]xi8>

  return %A : vector<8x[8]xi8>
}

// -----

// CHECK-LABEL: @negative_test_vector_type_not_supported
// CHECK-NOT: memref.collapse

func.func @negative_test_vector_type_not_supported(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[8]x[8]x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<?x?x?x8xi8>, vector<[8]x[8]x8xi8>

  return %A : vector<[8]x[8]x8xi8>
}

// -----

// CHECK-LABEL: @negative_test_non_mem
// CHECK-NOT: memref.collapse

func.func @negative_test_non_mem(%i : index, %j : index, %M : tensor<?x?x?x8xi8>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : tensor<?x?x?x8xi8>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// CHECK-LABEL: @negative_test_discontig_mem_0
// CHECK-NOT: memref.collapse

#s3 = strided<[?, ?, 16, 1]>

func.func @negative_test_discontig_mem_0(%i : index, %j : index, %M : memref<?x?x?x8xi8, #s3>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?x8xi8, #s3>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// CHECK-LABEL: @negative_test_discontig_mem_1
// CHECK-NOT: memref.collapse

#layout = affine_map<(i, j, k, p) -> (j, i, k, p)>

func.func @negative_test_discontig_mem_1(%i : index, %j : index, %M : memref<?x?x?x8xi8, #layout>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?x8xi8, #layout>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}

// -----

// CHECK-LABEL: @negative_test_discontig_read_strided_vec
// CHECK-NOT: memref.collapse

func.func @negative_test_discontig_read_strided_vec(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[4]x4xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?x8xi8>, vector<[4]x4xi8>

  return %A : vector<[4]x4xi8>
}

// -----

// CHECK-LABEL: @negative_test_bcast_transp
// CHECK-NOT: memref.collapse

#perm = affine_map<(i, j, k, p) -> (k, 0)>

func.func @negative_test_bcast_transp(%i : index, %j : index, %M : memref<?x?x?x8xi8>) -> vector<[4]x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %A = vector.transfer_read %M[%i, %j, %c0, %c0], %c0_i8 {permutation_map = #perm, in_bounds = [true, true] } : memref<?x?x?x8xi8>, vector<[4]x8xi8>

  return %A : vector<[4]x8xi8>
}
