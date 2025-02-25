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

// CHECK-LABEL: @scalable_gather_memref_2d
// CHECK-SAME:      %[[BASE:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[IDXVEC:.*]]: vector<2x[3]xindex>,
// CHECK-SAME:      %[[MASK:.*]]: vector<2x[3]xi1>,
// CHECK-SAME:      %[[PASS:.*]]: vector<2x[3]xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x[3]xf32>
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

// Check that vector.gather of a strided memref is replaced with a
// vector.gather with indices encoding the original strides. Note that multiple
// patterns are run for this example, e.g.:
  //  1. "remove stride from gather source"
  //  2. "flatten gather"
// However, the main goal is to the test Pattern 1 above.
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
  %res = vector.gather %subview[%c0] [%idxs], %mask, %pass_thru : memref<100xf32, strided<[3]>>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %res : vector<4xf32>
}
// CHECK-LABEL:   func.func @strided_gather(
// CHECK-SAME:                         %[[base:.*]]: memref<100x3xf32>,
// CHECK-SAME:                         %[[IDXS:.*]]: vector<4xindex>,
// CHECK-SAME:                         %[[VAL_4:.*]]: index,
// CHECK-SAME:                         %[[VAL_5:.*]]: index) -> vector<4xf32> {
// CHECK:           %[[CST_3:.*]] = arith.constant dense<3> : vector<4xindex>
// CHECK:           %[[MASK:.*]] = arith.constant dense<true> : vector<4xi1>

// CHECK:           %[[COLLAPSED:.*]] = memref.collapse_shape %[[base]] {{\[\[}}0, 1]] : memref<100x3xf32> into memref<300xf32>
// CHECK:           %[[NEW_IDXS:.*]] = arith.muli %[[IDXS]], %[[CST_3]] : vector<4xindex>

// CHECK:           %[[MASK_0:.*]] = vector.extract %[[MASK]][0] : i1 from vector<4xi1>
// CHECK:           %[[IDX_0:.*]] = vector.extract %[[NEW_IDXS]][0] : index from vector<4xindex>
// CHECK:           scf.if %[[MASK_0]] -> (vector<4xf32>)
// CHECK:             %[[M_0:.*]] = vector.load %[[COLLAPSED]][%[[IDX_0]]] : memref<300xf32>, vector<1xf32>
// CHECK:             %[[V_0:.*]] = vector.extract %[[M_0]][0] : f32 from vector<1xf32>

// CHECK:           %[[MASK_1:.*]] = vector.extract %[[MASK]][1] : i1 from vector<4xi1>
// CHECK:           %[[IDX_1:.*]] = vector.extract %[[NEW_IDXS]][1] : index from vector<4xindex>
// CHECK:           scf.if %[[MASK_1]] -> (vector<4xf32>)
// CHECK:             %[[M_1:.*]] = vector.load %[[COLLAPSED]][%[[IDX_1]]] : memref<300xf32>, vector<1xf32>
// CHECK:             %[[V_1:.*]] = vector.extract %[[M_1]][0] : f32 from vector<1xf32>

// CHECK:           %[[MASK_2:.*]] = vector.extract %[[MASK]][2] : i1 from vector<4xi1>
// CHECK:           %[[IDX_2:.*]] = vector.extract %[[NEW_IDXS]][2] : index from vector<4xindex>
// CHECK:           scf.if %[[MASK_2]] -> (vector<4xf32>)
// CHECK:             %[[M_2:.*]] = vector.load %[[COLLAPSED]][%[[IDX_2]]] : memref<300xf32>, vector<1xf32>
// CHECK:             %[[V_2:.*]] = vector.extract %[[M_2]][0] : f32 from vector<1xf32>

// CHECK:           %[[MASK_3:.*]] = vector.extract %[[MASK]][3] : i1 from vector<4xi1>
// CHECK:           %[[IDX_3:.*]] = vector.extract %[[NEW_IDXS]][3] : index from vector<4xindex>
// CHECK:           scf.if %[[MASK_3]] -> (vector<4xf32>)
// CHECK:             %[[M_3:.*]] = vector.load %[[COLLAPSED]][%[[IDX_3]]] : memref<300xf32>, vector<1xf32>
// CHECK:             %[[V_3:.*]] = vector.extract %[[M_3]][0] : f32 from vector<1xf32>

// CHECK-LABEL: @scalable_gather_1d
// CHECK-NOT: extract
// CHECK: vector.gather
// CHECK-NOT: extract
func.func @scalable_gather_1d(%base: tensor<?xf32>, %v: vector<[2]xindex>, %mask: vector<[2]xi1>, %pass_thru: vector<[2]xf32>) -> vector<[2]xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : tensor<?xf32>, vector<[2]xindex>, vector<[2]xi1>, vector<[2]xf32> into vector<[2]xf32>
  return %0 : vector<[2]xf32>
}
