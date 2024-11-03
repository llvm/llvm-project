// RUN: mlir-opt %s --test-vector-gather-lowering | FileCheck %s
// RUN: mlir-opt %s --test-vector-gather-lowering --canonicalize | FileCheck %s --check-prefix=CANON

// CHECK-LABEL: @gather_memref_1d
// CHECK-SAME:    ([[BASE:%.+]]: memref<?xf32>, [[IDXVEC:%.+]]: vector<2xindex>, [[MASK:%.+]]: vector<2xi1>, [[PASS:%.+]]: vector<2xf32>)
// CHECK-DAG:     [[M0:%.+]]    = vector.extract [[MASK]][0] : i1 from vector<2xi1>
// CHECK-DAG:     %[[IDX0:.+]]  = vector.extract [[IDXVEC]][0] : index from vector<2xindex>
// CHECK-NEXT:    [[RES0:%.+]]  = scf.if [[M0]] -> (vector<2xf32>)
// CHECK-NEXT:      [[LD0:%.+]]   = vector.load [[BASE]][%[[IDX0]]] : memref<?xf32>, vector<1xf32>
// CHECK-NEXT:      [[ELEM0:%.+]] = vector.extract [[LD0]][0] : f32 from vector<1xf32>
// CHECK-NEXT:      [[INS0:%.+]]  = vector.insert [[ELEM0]], [[PASS]] [0] : f32 into vector<2xf32>
// CHECK-NEXT:      scf.yield [[INS0]] : vector<2xf32>
// CHECK-NEXT:    else
// CHECK-NEXT:      scf.yield [[PASS]] : vector<2xf32>
// CHECK-DAG:     [[M1:%.+]]    = vector.extract [[MASK]][1] : i1 from vector<2xi1>
// CHECK-DAG:     %[[IDX1:.+]]  = vector.extract [[IDXVEC]][1] : index from vector<2xindex>
// CHECK-NEXT:    [[RES1:%.+]]  = scf.if [[M1]] -> (vector<2xf32>)
// CHECK-NEXT:      [[LD1:%.+]]   = vector.load [[BASE]][%[[IDX1]]] : memref<?xf32>, vector<1xf32>
// CHECK-NEXT:      [[ELEM1:%.+]] = vector.extract [[LD1]][0] : f32 from vector<1xf32>
// CHECK-NEXT:      [[INS1:%.+]]  = vector.insert [[ELEM1]], [[RES0]] [1] : f32 into vector<2xf32>
// CHECK-NEXT:      scf.yield [[INS1]] : vector<2xf32>
// CHECK-NEXT:    else
// CHECK-NEXT:      scf.yield [[RES0]] : vector<2xf32>
// CHECK:         return [[RES1]] : vector<2xf32>
func.func @gather_memref_1d(%base: memref<?xf32>, %v: vector<2xindex>, %mask: vector<2xi1>, %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : memref<?xf32>, vector<2xindex>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @gather_memref_1d_i32_index
// CHECK-SAME:    ([[BASE:%.+]]: memref<?xf32>, [[IDXVEC:%.+]]: vector<2xi32>, [[MASK:%.+]]: vector<2xi1>, [[PASS:%.+]]: vector<2xf32>)
// CHECK-DAG:     [[C42:%.+]]   = arith.constant 42 : index
// CHECK-DAG:     [[IDXS:%.+]]  = arith.index_cast [[IDXVEC]] : vector<2xi32> to vector<2xindex>
// CHECK-DAG:     [[IDX0:%.+]]  = vector.extract [[IDXS]][0] : index from vector<2xindex>
// CHECK-NEXT:    %[[OFF0:.+]]  = arith.addi [[IDX0]], [[C42]] : index
// CHECK-NEXT:    [[RES0:%.+]]  = scf.if
// CHECK-NEXT:      [[LD0:%.+]]   = vector.load [[BASE]][%[[OFF0]]] : memref<?xf32>, vector<1xf32>
// CHECK:         else
// CHECK:         [[IDX1:%.+]]  = vector.extract [[IDXS]][1] : index from vector<2xindex>
// CHECK:         %[[OFF1:.+]]  = arith.addi [[IDX1]], [[C42]] : index
// CHECK:         [[RES1:%.+]]  = scf.if
// CHECK-NEXT:      [[LD1:%.+]]   = vector.load [[BASE]][%[[OFF1]]] : memref<?xf32>, vector<1xf32>
// CHECK:         else
// CHECK:         return [[RES1]] : vector<2xf32>
func.func @gather_memref_1d_i32_index(%base: memref<?xf32>, %v: vector<2xi32>, %mask: vector<2xi1>, %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 42 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : memref<?xf32>, vector<2xi32>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @gather_memref_2d
// CHECK-SAME:    ([[BASE:%.+]]: memref<?x?xf32>, [[IDXVEC:%.+]]: vector<2x3xindex>, [[MASK:%.+]]: vector<2x3xi1>, [[PASS:%.+]]: vector<2x3xf32>)
// CHECK-DAG:     %[[C0:.+]]    = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]]    = arith.constant 1 : index
// CHECK-DAG:     [[PTV0:%.+]]  = vector.extract [[PASS]][0] : vector<3xf32> from vector<2x3xf32>
// CHECK-DAG:     [[M0:%.+]]    = vector.extract [[MASK]][0, 0] : i1 from vector<2x3xi1>
// CHECK-DAG:     [[IDX0:%.+]]  = vector.extract [[IDXVEC]][0, 0] : index from vector<2x3xindex>
// CHECK-NEXT:    %[[OFF0:.+]]  = arith.addi [[IDX0]], %[[C1]] : index
// CHECK-NEXT:    [[RES0:%.+]]  = scf.if [[M0]] -> (vector<3xf32>)
// CHECK-NEXT:      [[LD0:%.+]]   = vector.load [[BASE]][%[[C0]], %[[OFF0]]] : memref<?x?xf32>, vector<1xf32>
// CHECK-NEXT:      [[ELEM0:%.+]] = vector.extract [[LD0]][0] : f32 from vector<1xf32>
// CHECK-NEXT:      [[INS0:%.+]]  = vector.insert [[ELEM0]], [[PTV0]] [0] : f32 into vector<3xf32>
// CHECK-NEXT:      scf.yield [[INS0]] : vector<3xf32>
// CHECK-NEXT:    else
// CHECK-NEXT:      scf.yield [[PTV0]] : vector<3xf32>
// CHECK-COUNT-5: scf.if
// CHECK:         [[FINAL:%.+]] = vector.insert %{{.+}}, %{{.+}} [1] : vector<3xf32> into vector<2x3xf32>
// CHECK-NEXT:    return [[FINAL]] : vector<2x3xf32>
 func.func @gather_memref_2d(%base: memref<?x?xf32>, %v: vector<2x3xindex>, %mask: vector<2x3xi1>, %pass_thru: vector<2x3xf32>) -> vector<2x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru : memref<?x?xf32>, vector<2x3xindex>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
 }

// CHECK-LABEL: @gather_tensor_1d
// CHECK-SAME:    ([[BASE:%.+]]: tensor<?xf32>, [[IDXVEC:%.+]]: vector<2xindex>, [[MASK:%.+]]: vector<2xi1>, [[PASS:%.+]]: vector<2xf32>)
// CHECK-DAG:     [[M0:%.+]]    = vector.extract [[MASK]][0] : i1 from vector<2xi1>
// CHECK-DAG:     %[[IDX0:.+]]  = vector.extract [[IDXVEC]][0] : index from vector<2xindex>
// CHECK-NEXT:    [[RES0:%.+]]  = scf.if [[M0]] -> (vector<2xf32>)
// CHECK-NEXT:      [[ELEM0:%.+]] = tensor.extract [[BASE]][%[[IDX0]]] : tensor<?xf32>
// CHECK-NEXT:      [[INS0:%.+]]  = vector.insert [[ELEM0]], [[PASS]] [0] : f32 into vector<2xf32>
// CHECK-NEXT:      scf.yield [[INS0]] : vector<2xf32>
// CHECK-NEXT:    else
// CHECK-NEXT:      scf.yield [[PASS]] : vector<2xf32>
// CHECK-DAG:     [[M1:%.+]]    = vector.extract [[MASK]][1] : i1 from vector<2xi1>
// CHECK-DAG:     %[[IDX1:.+]]  = vector.extract [[IDXVEC]][1] : index from vector<2xindex>
// CHECK-NEXT:    [[RES1:%.+]]  = scf.if [[M1]] -> (vector<2xf32>)
// CHECK-NEXT:      [[ELEM1:%.+]] = tensor.extract [[BASE]][%[[IDX1]]] : tensor<?xf32>
// CHECK-NEXT:      [[INS1:%.+]]  = vector.insert [[ELEM1]], [[RES0]] [1] : f32 into vector<2xf32>
// CHECK-NEXT:      scf.yield [[INS1]] : vector<2xf32>
// CHECK-NEXT:    else
// CHECK-NEXT:      scf.yield [[RES0]] : vector<2xf32>
// CHECK:         return [[RES1]] : vector<2xf32>
func.func @gather_tensor_1d(%base: tensor<?xf32>, %v: vector<2xindex>, %mask: vector<2xi1>, %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : tensor<?xf32>, vector<2xindex>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @gather_tensor_2d
// CHECK:  scf.if
// CHECK:    tensor.extract
// CHECK:  else
// CHECK:  scf.if
// CHECK:    tensor.extract
// CHECK:  else
// CHECK:  scf.if
// CHECK:    tensor.extract
// CHECK:  else
// CHECK:  scf.if
// CHECK:    tensor.extract
// CHECK:  else
// CHECK:  scf.if
// CHECK:    tensor.extract
// CHECK:  else
// CHECK:  scf.if
// CHECK:    tensor.extract
// CHECK:  else
// CHECK:       [[FINAL:%.+]] = vector.insert %{{.+}}, %{{.+}} [1] : vector<3xf32> into vector<2x3xf32>
// CHECK-NEXT:  return [[FINAL]] : vector<2x3xf32>
 func.func @gather_tensor_2d(%base: tensor<?x?xf32>, %v: vector<2x3xindex>, %mask: vector<2x3xi1>, %pass_thru: vector<2x3xf32>) -> vector<2x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru : tensor<?x?xf32>, vector<2x3xindex>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
 }

// Check that all-set and no-set maskes get optimized out after canonicalization.

// CANON-LABEL: @gather_tensor_1d_all_set
// CANON-NOT:     scf.if
// CANON:         tensor.extract
// CANON:         tensor.extract
// CANON:         [[FINAL:%.+]] = vector.insert %{{.+}}, %{{.+}} [1] : f32 into vector<2xf32>
// CANON-NEXT:    return [[FINAL]] : vector<2xf32>
func.func @gather_tensor_1d_all_set(%base: tensor<?xf32>, %v: vector<2xindex>, %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %mask = arith.constant dense <true> : vector<2xi1>
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : tensor<?xf32>, vector<2xindex>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CANON-LABEL: @gather_tensor_1d_none_set
// CANON-SAME:    ([[BASE:%.+]]: tensor<?xf32>, [[IDXVEC:%.+]]: vector<2xindex>, [[PASS:%.+]]: vector<2xf32>)
// CANON-NEXT:    return [[PASS]] : vector<2xf32>
func.func @gather_tensor_1d_none_set(%base: tensor<?xf32>, %v: vector<2xindex>, %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %mask = arith.constant dense <false> : vector<2xi1>
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : tensor<?xf32>, vector<2xindex>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}
