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
// CHECK:         %[[OFF0:.+]]  = arith.addi [[IDX0]], %[[C1]] : index
// CHECK:         [[RES0:%.+]]  = scf.if [[M0]] -> (vector<3xf32>)
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

// CHECK-LABEL: @scalable_gather_memref_2d
// CHECK-SAME:      %[[BASE:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[IDXVEC:.*]]: vector<2x[3]xindex>,
// CHECK-SAME:      %[[MASK:.*]]: vector<2x[3]xi1>,
// CHECK-SAME:      %[[PASS:.*]]: vector<2x[3]xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[INIT:.*]] = ub.poison : vector<2x[3]xf32>
// CHECK:         %[[IDXVEC0:.*]] = vector.extract %[[IDXVEC]][0] : vector<[3]xindex> from vector<2x[3]xindex>
// CHECK:         %[[MASK0:.*]] = vector.extract %[[MASK]][0] : vector<[3]xi1> from vector<2x[3]xi1>
// CHECK:         %[[PASS0:.*]] = vector.extract %[[PASS]][0] : vector<[3]xf32> from vector<2x[3]xf32>
// CHECK:         %[[GATHER0:.*]] = vector.gather %[[BASE]]{{\[}}%[[C0]], %[[C1]]] {{\[}}%[[IDXVEC0]]], %[[MASK0]], %[[PASS0]] : memref<?x?xf32>, vector<[3]xindex>, vector<[3]xi1>, vector<[3]xf32> into vector<[3]xf32>
// CHECK:         %[[INS0:.*]] = vector.insert %[[GATHER0]], %[[INIT]] [0] : vector<[3]xf32> into vector<2x[3]xf32>
// CHECK:         %[[IDXVEC1:.*]] = vector.extract %[[IDXVEC]][1] : vector<[3]xindex> from vector<2x[3]xindex>
// CHECK:         %[[MASK1:.*]] = vector.extract %[[MASK]][1] : vector<[3]xi1> from vector<2x[3]xi1>
// CHECK:         %[[PASS1:.*]] = vector.extract %[[PASS]][1] : vector<[3]xf32> from vector<2x[3]xf32>
// CHECK:         %[[GATHER1:.*]] = vector.gather %[[BASE]]{{\[}}%[[C0]], %[[C1]]] {{\[}}%[[IDXVEC1]]], %[[MASK1]], %[[PASS1]] : memref<?x?xf32>, vector<[3]xindex>, vector<[3]xi1>, vector<[3]xf32> into vector<[3]xf32>
// CHECK:         %[[INS1:.*]] = vector.insert %[[GATHER1]], %[[INS0]] [1] : vector<[3]xf32> into vector<2x[3]xf32>
// CHECK-NEXT:    return %[[INS1]] : vector<2x[3]xf32>
func.func @scalable_gather_memref_2d(%base: memref<?x?xf32>, %v: vector<2x[3]xindex>, %mask: vector<2x[3]xi1>, %pass_thru: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru : memref<?x?xf32>, vector<2x[3]xindex>, vector<2x[3]xi1>, vector<2x[3]xf32> into vector<2x[3]xf32>
 return %0 : vector<2x[3]xf32>
}

// CHECK-LABEL: @scalable_gather_memref_2d_with_alignment
// CHECK:         vector.gather
// CHECK-SAME:    {alignment = 8 : i64}
// CHECK:         vector.gather
// CHECK-SAME:    {alignment = 8 : i64}
func.func @scalable_gather_memref_2d_with_alignment(%base: memref<?x?xf32>, %v: vector<2x[3]xindex>, %mask: vector<2x[3]xi1>, %pass_thru: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru {alignment = 8} : memref<?x?xf32>, vector<2x[3]xindex>, vector<2x[3]xi1>, vector<2x[3]xf32> into vector<2x[3]xf32>
 return %0 : vector<2x[3]xf32>
}

// CHECK-LABEL: @scalable_gather_cant_unroll
// CHECK-NOT: extract
// CHECK: vector.gather
// CHECK-NOT: extract
func.func @scalable_gather_cant_unroll(%base: memref<?x?xf32>, %v: vector<[4]x8xindex>, %mask: vector<[4]x8xi1>, %pass_thru: vector<[4]x8xf32>) -> vector<[4]x8xf32> {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru : memref<?x?xf32>, vector<[4]x8xindex>, vector<[4]x8xi1>, vector<[4]x8xf32> into vector<[4]x8xf32>
 return %0 : vector<[4]x8xf32>
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

// CHECK-LABEL: @gather_memref_non_unit_stride_read_1_element
// CHECK: %[[MASK:.*]] = vector.extract %arg2[0] : i1 from vector<1xi1>
// CHECK: %[[IDX:.*]] = vector.extract %arg1[0] : index from vector<1xindex>
// CHECK: %[[RET:.*]] = scf.if %[[MASK]] -> (vector<1xf32>) {
// CHECK:   %[[VEC:.*]] = vector.load %arg0[%[[IDX]]] : memref<4xf32, strided<[2]>>, vector<1xf32>
// CHECK:   %[[VAL:.*]] = vector.extract %[[VEC]][0] : f32 from vector<1xf32>
// CHECK:   %[[RES:.*]] = vector.insert %[[VAL]], %arg3 [0] : f32 into vector<1xf32>
// CHECK:   scf.yield %[[RES]] : vector<1xf32>
// CHECK: } else {
// CHECK:    scf.yield %arg3 : vector<1xf32>
// CHECK: }
// CHECK: return %[[RET]] : vector<1xf32>
func.func @gather_memref_non_unit_stride_read_1_element(%base: memref<4xf32, strided<[2]>>, %v: vector<1xindex>, %mask: vector<1xi1>, %pass_thru: vector<1xf32>) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : memref<4xf32, strided<[2]>>, vector<1xindex>, vector<1xi1>, vector<1xf32> into vector<1xf32>
  return %0 : vector<1xf32>
}

// CHECK-LABEL: @gather_memref_non_unit_stride_read_more_than_1_element
// CHECK: %[[CONST:.*]] = arith.constant 0 : index
// CHECK: %[[RET:.*]] = vector.gather %arg0[%[[CONST]]] [%arg1], %arg2, %arg3 : memref<4xf32, strided<[2]>>, vector<2xindex>, vector<2xi1>, vector<2xf32> into vector<2xf32>
// CHECK: return %[[RET]] : vector<2xf32>
func.func @gather_memref_non_unit_stride_read_more_than_1_element(%base: memref<4xf32, strided<[2]>>, %v: vector<2xindex>, %mask: vector<2xi1>, %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : memref<4xf32, strided<[2]>>, vector<2xindex>, vector<2xi1>, vector<2xf32> into vector<2xf32>
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
// CANON:         [[FINAL:%.+]] = vector.from_elements %{{.+}}, %{{.+}} : vector<2xf32>
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

// Check that vector.gather of a rank-reducing strided memref.subview is
// replaced with a vector.gather on the source memref. The source gather gets
// an explicit zero index vector for the rank-reduced trailing dimension.
#map = affine_map<()[s0] -> (s0 * 4096)>
func.func @strided_gather(%base : memref<100x3xf32>,
                          %idxs : vector<4xindex>,
                          %x : index, %y : index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %x_1 = affine.apply #map()[%x]
  // Strided MemRef
  %subview = memref.subview %base[0, 0] [100, 1] [1, 1] : memref<100x3xf32> to memref<100xf32, strided<[3]>>
  %mask = arith.constant dense<true> : vector<4xi1>
  %pass_thru = arith.constant dense<0.000000e+00> : vector<4xf32>
  // Gather of a strided MemRef
  %res = vector.gather %subview[%c0] [%idxs], %mask, %pass_thru {alignment = 8} : memref<100xf32, strided<[3]>>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK-LABEL:   func.func @strided_gather(
// CHECK-SAME:                         %[[base:.*]]: memref<100x3xf32>,
// CHECK-SAME:                         %[[IDXS:.*]]: vector<4xindex>,
// CHECK-SAME:                         %[[VAL_4:.*]]: index,
// CHECK-SAME:                         %[[VAL_5:.*]]: index) -> vector<4xf32> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[TRUE:.*]] = arith.constant true

// CHECK-NOT:       arith.muli
// CHECK:           %[[IDX_0:.*]] = vector.extract %[[IDXS]][0] : index from vector<4xindex>
// CHECK:           scf.if %[[TRUE]] -> (vector<4xf32>)
// CHECK:             %[[M_0:.*]] = vector.load %[[base]][%[[IDX_0]], %[[C0]]] {alignment = 8 : i64} : memref<100x3xf32>, vector<1xf32>
// CHECK:             %[[V_0:.*]] = vector.extract %[[M_0]][0] : f32 from vector<1xf32>

// CHECK:           %[[IDX_1:.*]] = vector.extract %[[IDXS]][1] : index from vector<4xindex>
// CHECK:           scf.if %[[TRUE]] -> (vector<4xf32>)
// CHECK:             %[[M_1:.*]] = vector.load %[[base]][%[[IDX_1]], %[[C0]]] {alignment = 8 : i64} : memref<100x3xf32>, vector<1xf32>
// CHECK:             %[[V_1:.*]] = vector.extract %[[M_1]][0] : f32 from vector<1xf32>

// CHECK:           %[[IDX_2:.*]] = vector.extract %[[IDXS]][2] : index from vector<4xindex>
// CHECK:           scf.if %[[TRUE]] -> (vector<4xf32>)
// CHECK:             %[[M_2:.*]] = vector.load %[[base]][%[[IDX_2]], %[[C0]]] {alignment = 8 : i64} : memref<100x3xf32>, vector<1xf32>
// CHECK:             %[[V_2:.*]] = vector.extract %[[M_2]][0] : f32 from vector<1xf32>

// CHECK:           %[[IDX_3:.*]] = vector.extract %[[IDXS]][3] : index from vector<4xindex>
// CHECK:           scf.if %[[TRUE]] -> (vector<4xf32>)
// CHECK:             %[[M_3:.*]] = vector.load %[[base]][%[[IDX_3]], %[[C0]]] {alignment = 8 : i64} : memref<100x3xf32>, vector<1xf32>
// CHECK:             %[[V_3:.*]] = vector.extract %[[M_3]][0] : f32 from vector<1xf32>

// Same as @strided_gather but with non-zero subview offsets. Both subview
// offsets must end up in the lowered loads: the outer one composed into the
// first index, the inner one into the second.
func.func @strided_gather_nonzero_subview_offsets(
    %base: memref<100x3xf32>, %idxs: vector<4xindex>, %row: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %subview = memref.subview %base[%row, %c2][95, 1][1, 1]
      : memref<100x3xf32> to memref<95xf32, strided<[3], offset: ?>>
  %mask = arith.constant dense<true> : vector<4xi1>
  %pass = arith.constant dense<0.000000e+00> : vector<4xf32>
  %res = vector.gather %subview[%c0] [%idxs], %mask, %pass
      : memref<95xf32, strided<[3], offset: ?>>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK-LABEL: func.func @strided_gather_nonzero_subview_offsets(
// CHECK-SAME:    %[[BASE:.*]]: memref<100x3xf32>,
// CHECK-SAME:    %[[IDXS:.*]]: vector<4xindex>,
// CHECK-SAME:    %[[ROW:.*]]: index)
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[I0:.*]] = vector.extract %[[IDXS]][0]
// CHECK:         %[[O0:.*]] = arith.addi %[[ROW]], %[[I0]]
// CHECK:         vector.load %[[BASE]][%[[O0]], %[[C2]]]
// CHECK:         %[[I1:.*]] = vector.extract %[[IDXS]][1]
// CHECK:         %[[O1:.*]] = arith.addi %[[ROW]], %[[I1]]
// CHECK:         vector.load %[[BASE]][%[[O1]], %[[C2]]]

// Dynamic outer subview stride: the rewrite splats the dynamic stride across
// the index vector and multiplies the original indices by it before lowering
// to the 2-D gather, so no `subview` survives in the output.
func.func @strided_gather_dynamic_subview_stride(
    %base: memref<100x3xf32>, %idxs: vector<4xindex>, %s: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %subview = memref.subview %base[0, 0][100, 1][%s, 1]
      : memref<100x3xf32> to memref<100xf32, strided<[?]>>
  %mask = arith.constant dense<true> : vector<4xi1>
  %pass = arith.constant dense<0.000000e+00> : vector<4xf32>
  %res = vector.gather %subview[%c0] [%idxs], %mask, %pass
      : memref<100xf32, strided<[?]>>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK-LABEL: func.func @strided_gather_dynamic_subview_stride(
// CHECK-SAME:    %[[BASE:.*]]: memref<100x3xf32>,
// CHECK-SAME:    %[[IDXS:.*]]: vector<4xindex>,
// CHECK-SAME:    %[[S:.*]]: index)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[BCAST:.*]] = vector.broadcast %[[S]] : index to vector<4xindex>
// CHECK:         %[[STRIDED:.*]] = arith.muli %[[IDXS]], %[[BCAST]] : vector<4xindex>
// CHECK-NOT:     memref.subview
// CHECK:         %[[I0:.*]] = vector.extract %[[STRIDED]][0]
// CHECK:         vector.load %[[BASE]][%[[I0]], %[[C0]]]
// CHECK:         %[[I1:.*]] = vector.extract %[[STRIDED]][1]
// CHECK:         vector.load %[[BASE]][%[[I1]], %[[C0]]]

// Static non-unit outer subview stride: same shape as the dynamic case, but
// the multiply is by a constant splat.
func.func @strided_gather_static_outer_stride(
    %base: memref<100x3xf32>, %idxs: vector<4xindex>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %subview = memref.subview %base[0, 0][50, 1][2, 1]
      : memref<100x3xf32> to memref<50xf32, strided<[6]>>
  %mask = arith.constant dense<true> : vector<4xi1>
  %pass = arith.constant dense<0.000000e+00> : vector<4xf32>
  %res = vector.gather %subview[%c0] [%idxs], %mask, %pass
      : memref<50xf32, strided<[6]>>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK-LABEL: func.func @strided_gather_static_outer_stride(
// CHECK-SAME:    %[[BASE:.*]]: memref<100x3xf32>,
// CHECK-SAME:    %[[IDXS:.*]]: vector<4xindex>)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[CST_2:.*]] = arith.constant dense<2> : vector<4xindex>
// CHECK:         %[[STRIDED:.*]] = arith.muli %[[IDXS]], %[[CST_2]] : vector<4xindex>
// CHECK-NOT:     memref.subview
// CHECK:         %[[I0:.*]] = vector.extract %[[STRIDED]][0]
// CHECK:         vector.load %[[BASE]][%[[I0]], %[[C0]]]
// CHECK:         %[[I1:.*]] = vector.extract %[[STRIDED]][1]
// CHECK:         vector.load %[[BASE]][%[[I1]], %[[C0]]]

// Regression: with a non-unit outer stride AND a non-zero gather offset, the
// gather's offset is in subview-element units, so it must be scaled by the
// outer stride before adding to the subview's outer offset. Previously, this
// pattern silently dropped the scaling.
func.func @strided_gather_static_stride_nonzero_offset(
    %base: memref<100x3xf32>, %idxs: vector<4xindex>) -> vector<4xf32> {
  %c5 = arith.constant 5 : index
  %subview = memref.subview %base[0, 0][50, 1][2, 1]
      : memref<100x3xf32> to memref<50xf32, strided<[6]>>
  %mask = arith.constant dense<true> : vector<4xi1>
  %pass = arith.constant dense<0.000000e+00> : vector<4xf32>
  %res = vector.gather %subview[%c5] [%idxs], %mask, %pass
      : memref<50xf32, strided<[6]>>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK-LABEL: func.func @strided_gather_static_stride_nonzero_offset(
// CHECK-SAME:    %[[BASE:.*]]: memref<100x3xf32>,
// CHECK-SAME:    %[[IDXS:.*]]: vector<4xindex>)
// `5 * 2 = 10` source rows skipped at the gather offset, plus the outer
// subview offset of 0.
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK:         %[[I0:.*]] = vector.extract %{{.*}}[0]
// CHECK:         %[[O0:.*]] = arith.addi %[[I0]], %[[C10]]
// CHECK:         vector.load %[[BASE]][%[[O0]], %{{.*}}]

// Verify that the subview-to-source rewrite handles N-D source memrefs. Both
// kept source dimensions are multiplied by their subview strides, while the two
// rank-reduced dimensions are represented as zero index vectors and therefore
// lower to their composed scalar offsets.
func.func @strided_gather_nd_rank_reduced(
    %base: memref<8x9x10x11xf32>,
    %idx0: vector<2xindex>, %idx1: vector<2xindex>,
    %off0: index, %off1: index, %off2: index, %off3: index,
    %gather_off0: index, %gather_off1: index,
    %stride0: index, %stride2: index) -> vector<2xf32> {
  %subview = memref.subview
      %base[%off0, %off1, %off2, %off3][4, 1, 5, 1][%stride0, 1, %stride2, 1]
      : memref<8x9x10x11xf32> to memref<4x5xf32, strided<[?, ?], offset: ?>>
  %mask = arith.constant dense<true> : vector<2xi1>
  %pass = arith.constant dense<0.000000e+00> : vector<2xf32>
  %res = vector.gather %subview[%gather_off0, %gather_off1] [%idx0, %idx1],
      %mask, %pass
      : memref<4x5xf32, strided<[?, ?], offset: ?>>, vector<2xindex>,
        vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %res : vector<2xf32>
}
// CHECK-LABEL: func.func @strided_gather_nd_rank_reduced(
// CHECK-SAME:    %[[BASE:[^:]+]]: memref<8x9x10x11xf32>,
// CHECK-SAME:    %[[IDX0:[^:]+]]: vector<2xindex>, %[[IDX1:[^:]+]]: vector<2xindex>,
// CHECK-SAME:    %[[OFF0:[^:]+]]: index, %[[OFF1:[^:]+]]: index, %[[OFF2:[^:]+]]: index, %[[OFF3:[^:]+]]: index,
// CHECK-SAME:    %[[GATHER_OFF0:[^:]+]]: index, %[[GATHER_OFF1:[^:]+]]: index,
// CHECK-SAME:    %[[STRIDE0:[^:]+]]: index, %[[STRIDE2:[^:]+]]: index)
// CHECK:         %[[SCALED_OFF0:.*]] = arith.muli %[[GATHER_OFF0]], %[[STRIDE0]] : index
// CHECK:         %[[BASE_OFF0:.*]] = arith.addi %[[OFF0]], %[[SCALED_OFF0]] : index
// CHECK:         %[[BCAST0:.*]] = vector.broadcast %[[STRIDE0]] : index to vector<2xindex>
// CHECK:         %[[STRIDED_IDX0:.*]] = arith.muli %[[IDX0]], %[[BCAST0]] : vector<2xindex>
// CHECK:         %[[SCALED_OFF2:.*]] = arith.muli %[[GATHER_OFF1]], %[[STRIDE2]] : index
// CHECK:         %[[BASE_OFF2:.*]] = arith.addi %[[OFF2]], %[[SCALED_OFF2]] : index
// CHECK:         %[[BCAST2:.*]] = vector.broadcast %[[STRIDE2]] : index to vector<2xindex>
// CHECK:         %[[STRIDED_IDX1:.*]] = arith.muli %[[IDX1]], %[[BCAST2]] : vector<2xindex>
// CHECK:         %[[I0_0:.*]] = vector.extract %[[STRIDED_IDX0]][0]
// CHECK:         %[[LOAD_OFF0_0:.*]] = arith.addi %[[BASE_OFF0]], %[[I0_0]]
// CHECK:         %[[I1_0:.*]] = vector.extract %[[STRIDED_IDX1]][0]
// CHECK:         %[[LOAD_OFF2_0:.*]] = arith.addi %[[BASE_OFF2]], %[[I1_0]]
// CHECK:         vector.load %[[BASE]][%[[LOAD_OFF0_0]], %[[OFF1]], %[[LOAD_OFF2_0]], %[[OFF3]]]

func.func @strided_gather_nd_rank_reduced_scalable(
    %base: memref<8x9x10x11xf32>,
    %idx0: vector<[2]xindex>, %idx1: vector<[2]xindex>,
    %off0: index, %off1: index, %off2: index, %off3: index,
    %gather_off0: index, %gather_off1: index,
    %stride0: index, %stride2: index,
    %mask: vector<[2]xi1>, %pass: vector<[2]xf32>) -> vector<[2]xf32> {
  %subview = memref.subview
      %base[%off0, %off1, %off2, %off3][4, 1, 5, 1][%stride0, 1, %stride2, 1]
      : memref<8x9x10x11xf32> to memref<4x5xf32, strided<[?, ?], offset: ?>>
  %res = vector.gather %subview[%gather_off0, %gather_off1] [%idx0, %idx1],
      %mask, %pass
      : memref<4x5xf32, strided<[?, ?], offset: ?>>, vector<[2]xindex>,
        vector<[2]xi1>, vector<[2]xf32> into vector<[2]xf32>
  return %res : vector<[2]xf32>
}
// CHECK-LABEL: func.func @strided_gather_nd_rank_reduced_scalable(
// CHECK-SAME:    %[[BASE:[^:]+]]: memref<8x9x10x11xf32>,
// CHECK-SAME:    %[[IDX0:[^:]+]]: vector<[2]xindex>, %[[IDX1:[^:]+]]: vector<[2]xindex>,
// CHECK-SAME:    %[[OFF0:[^:]+]]: index, %[[OFF1:[^:]+]]: index, %[[OFF2:[^:]+]]: index, %[[OFF3:[^:]+]]: index,
// CHECK-SAME:    %[[GATHER_OFF0:[^:]+]]: index, %[[GATHER_OFF1:[^:]+]]: index,
// CHECK-SAME:    %[[STRIDE0:[^:]+]]: index, %[[STRIDE2:[^:]+]]: index,
// CHECK-SAME:    %[[MASK:[^:]+]]: vector<[2]xi1>, %[[PASS:[^:]+]]: vector<[2]xf32>)
// CHECK-NOT:     memref.collapse_shape
// CHECK:         %[[ZERO:.*]] = arith.constant dense<0> : vector<[2]xindex>
// CHECK:         %[[SCALED_OFF0:.*]] = arith.muli %[[GATHER_OFF0]], %[[STRIDE0]] : index
// CHECK:         %[[BASE_OFF0:.*]] = arith.addi %[[OFF0]], %[[SCALED_OFF0]] : index
// CHECK:         %[[BCAST0:.*]] = vector.broadcast %[[STRIDE0]] : index to vector<[2]xindex>
// CHECK:         %[[STRIDED_IDX0:.*]] = arith.muli %[[IDX0]], %[[BCAST0]] : vector<[2]xindex>
// CHECK:         %[[SCALED_OFF2:.*]] = arith.muli %[[GATHER_OFF1]], %[[STRIDE2]] : index
// CHECK:         %[[BASE_OFF2:.*]] = arith.addi %[[OFF2]], %[[SCALED_OFF2]] : index
// CHECK:         %[[BCAST2:.*]] = vector.broadcast %[[STRIDE2]] : index to vector<[2]xindex>
// CHECK:         %[[STRIDED_IDX1:.*]] = arith.muli %[[IDX1]], %[[BCAST2]] : vector<[2]xindex>
// CHECK:         %[[GATHER:.*]] = vector.gather %[[BASE]][%[[BASE_OFF0]], %[[OFF1]], %[[BASE_OFF2]], %[[OFF3]]] [%[[STRIDED_IDX0]], %[[ZERO]], %[[STRIDED_IDX1]], %[[ZERO]]], %[[MASK]], %[[PASS]]
// CHECK-SAME:      : memref<8x9x10x11xf32>, vector<[2]xindex>, vector<[2]xi1>, vector<[2]xf32> into vector<[2]xf32>
// CHECK:         return %[[GATHER]] : vector<[2]xf32>

// Verify that multi-index gather on a 2D memref correctly offsets each
// dimension independently.
// CHECK-LABEL: @gather_memref_2d_multi_index
// CHECK-SAME:    (%[[BASE:.+]]: memref<?x?xf32>,
// CHECK-SAME:     %[[IDX0:.+]]: vector<2xindex>, %[[IDX1:.+]]: vector<2xindex>,
// CHECK-SAME:     %[[MASK:.+]]: vector<2xi1>, %[[PASS:.+]]: vector<2xf32>)
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[M0:.+]] = vector.extract %[[MASK]][0]
// CHECK:         %[[I0_0:.+]] = vector.extract %[[IDX0]][0]
// CHECK:         %[[I1_0:.+]] = vector.extract %[[IDX1]][0]
// CHECK:         %[[OFF1_0:.+]] = arith.addi %[[I1_0]], %[[C1]]
// CHECK:         scf.if %[[M0]]
// CHECK:           vector.load %[[BASE]][%[[I0_0]], %[[OFF1_0]]]
func.func @gather_memref_2d_multi_index(
    %base: memref<?x?xf32>,
    %idx0: vector<2xindex>, %idx1: vector<2xindex>,
    %mask: vector<2xi1>, %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.gather %base[%c0, %c1][%idx0, %idx1], %mask, %pass_thru
    : memref<?x?xf32>, vector<2xindex>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @scalable_gather_1d
// CHECK-NOT: extract
// CHECK: vector.gather
// CHECK-NOT: extract
func.func @scalable_gather_1d(%base: tensor<?xf32>, %v: vector<[2]xindex>, %mask: vector<[2]xi1>, %pass_thru: vector<[2]xf32>) -> vector<[2]xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : tensor<?xf32>, vector<[2]xindex>, vector<[2]xi1>, vector<[2]xf32> into vector<[2]xf32>
  return %0 : vector<[2]xf32>
}

// Verify that gather on a 2D memref with zero base offsets directly
// adds each index element to its corresponding offset.

// CHECK-LABEL: @gather_memref_2d_single_index
// CHECK-SAME:    (%[[BASE:.+]]: memref<4x2xf32>,
// CHECK-SAME:     %[[IDXVEC:.+]]: vector<4xi32>,
// CHECK-SAME:     %[[MASK:.+]]: vector<4xi1>,
// CHECK-SAME:     %[[PASS:.+]]: vector<4xf32>)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[IDXS:.+]] = arith.index_cast %[[IDXVEC]]
//
// CHECK:         %[[IDX0:.+]] = vector.extract %[[IDXS]][0]
// CHECK:         scf.if
// CHECK:           vector.load %[[BASE]][%[[C0]], %[[IDX0]]] : memref<4x2xf32>, vector<1xf32>
//
// CHECK:         %[[IDX1:.+]] = vector.extract %[[IDXS]][1]
// CHECK:         scf.if
// CHECK:           vector.load %[[BASE]][%{{.+}}, %[[IDX1]]] : memref<4x2xf32>, vector<1xf32>
//
// CHECK:         %[[IDX2:.+]] = vector.extract %[[IDXS]][2]
// CHECK:         scf.if
// CHECK:           vector.load %[[BASE]][%{{.+}}, %[[IDX2]]] : memref<4x2xf32>, vector<1xf32>
//
// CHECK:         %[[IDX3:.+]] = vector.extract %[[IDXS]][3]
// CHECK:         scf.if
// CHECK:           vector.load %[[BASE]][%{{.+}}, %[[IDX3]]] : memref<4x2xf32>, vector<1xf32>
func.func @gather_memref_2d_single_index(
    %base: memref<4x2xf32>,
    %v: vector<4xi32>, %mask: vector<4xi1>,
    %pass_thru: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0, %c0][%v], %mask, %pass_thru
    : memref<4x2xf32>, vector<4xi32>,
      vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// Verify that gather on a 2D memref with non-zero base offsets correctly
// adds each index element to the last-dimension offset.

// CHECK-LABEL: @gather_memref_2d_single_index_nonzero_offsets
// CHECK-SAME:    (%[[BASE:.+]]: memref<4x2xf32>,
// CHECK-SAME:     %[[OFF0:.+]]: index, %[[OFF1:.+]]: index,
// CHECK-SAME:     %[[IDXVEC:.+]]: vector<2xi32>,
// CHECK-SAME:     %[[MASK:.+]]: vector<2xi1>,
// CHECK-SAME:     %[[PASS:.+]]: vector<2xf32>)
// CHECK-DAG:     %[[IDXS:.+]] = arith.index_cast %[[IDXVEC]]
// CHECK:         %[[IDX0:.+]] = vector.extract %[[IDXS]][0]
// CHECK:         %[[SUM0:.+]] = arith.addi %[[OFF1]], %[[IDX0]]
// CHECK:         scf.if
// CHECK:           vector.load %[[BASE]][%[[OFF0]], %[[SUM0]]]
func.func @gather_memref_2d_single_index_nonzero_offsets(
    %base: memref<4x2xf32>,
    %off0: index, %off1: index,
    %v: vector<2xi32>, %mask: vector<2xi1>,
    %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.gather %base[%off0, %off1][%v], %mask, %pass_thru
    : memref<4x2xf32>, vector<2xi32>,
      vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// -----

// CHECK-LABEL:   func.func @strided_gather_with_non_zero_offset(
// CHECK-SAME:        %[[BASE:.+]]: memref<6x5xf32>,
// CHECK-SAME:        %[[IDXS:.+]]: vector<2xindex>
// CHECK-DAG:       %[[OFFSET:.+]] = arith.constant 3 : index
// CHECK:           %[[IDX_0:.+]] = vector.extract %[[IDXS]][0]
// CHECK:           scf.if
// CHECK:             vector.load %[[BASE]][%[[IDX_0]], %[[OFFSET]]] : memref<6x5xf32>
// CHECK:           %[[IDX_1:.+]] = vector.extract %[[IDXS]][1]
// CHECK:           scf.if
// CHECK:             vector.load %[[BASE]][%[[IDX_1]], %[[OFFSET]]] : memref<6x5xf32>
func.func @strided_gather_with_non_zero_offset(%base: memref<6x5xf32>,
                                               %idxs: vector<2xindex>,
                                               %mask: vector<2xi1>,
                                               %pass_thru: vector<2xf32>)
    -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %sub = memref.subview %base[0, 3] [6, 1] [1, 1]
    : memref<6x5xf32> to memref<6xf32, strided<[5], offset: 3>>
  %0 = vector.gather %sub[%c0] [%idxs], %mask, %pass_thru
    : memref<6xf32, strided<[5], offset: 3>>, vector<2xindex>,
      vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// -----

// Dynamic subview offsets are composed into the source gather offsets.
// CHECK-LABEL:   func.func @strided_gather_with_dynamic_subview_offset(
// CHECK-SAME:        %[[BASE:.+]]: memref<4x3xf32>,
// CHECK-SAME:        %[[COL:.+]]: index,
// CHECK-SAME:        %[[IDXS:.+]]: vector<2xindex>
// CHECK:           %[[IDX_0:.+]] = vector.extract %[[IDXS]][0]
// CHECK:           scf.if
// CHECK:             vector.load %[[BASE]][%[[IDX_0]], %[[COL]]] : memref<4x3xf32>
// CHECK:           %[[IDX_1:.+]] = vector.extract %[[IDXS]][1]
// CHECK:           scf.if
// CHECK:             vector.load %[[BASE]][%[[IDX_1]], %[[COL]]] : memref<4x3xf32>
func.func @strided_gather_with_dynamic_subview_offset(
    %base: memref<4x3xf32>,
    %col: index,
    %idxs: vector<2xindex>,
    %mask: vector<2xi1>,
    %pass_thru: vector<2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %sub = memref.subview %base[0, %col] [4, 1] [1, 1]
    : memref<4x3xf32> to memref<4xf32, strided<[3], offset: ?>>
  %0 = vector.gather %sub[%c0] [%idxs], %mask, %pass_thru
    : memref<4xf32, strided<[3], offset: ?>>, vector<2xindex>,
      vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}
