// RUN: mlir-opt %s -test-vector-contraction-lowering | FileCheck %s
// RUN: mlir-opt %s -test-vector-contraction-lowering=vector-lower-matrix-intrinsics=1 | FileCheck %s --check-prefix=MATRIX
// RUN: mlir-opt %s -test-vector-contraction-lowering=vector-outerproduct=1 | FileCheck %s --check-prefix=OUTERPRODUCT
// RUN: mlir-opt %s -test-vector-contraction-lowering=vector-filter-outerproduct=1 | FileCheck %s --check-prefix=FILTEROUTERPRODUCT
// RUN: mlir-opt %s -test-vector-contraction-lowering=vector-parallel-arith=1 | FileCheck %s --check-prefix=PARALLEL

#dotp_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dotp_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"]
}

// CHECK-LABEL: func @extract_contract1
// CHECK-SAME: %[[A:.*0]]: vector<4xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<4xf32>,
// CHECK-SAME: %[[C:.*2]]: f32
// CHECK:      %[[F:.*]] = arith.mulf %[[A]], %[[B]] : vector<4xf32>
// CHECK:      %[[R:.*]] = vector.reduction <add>, %[[F]], %[[C]] : vector<4xf32> into f32
// CHECK:      return %[[R]] : f32

func.func @extract_contract1(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: f32) -> f32 {
  %0 = vector.contract #dotp_trait %arg0, %arg1, %arg2
    : vector<4xf32>, vector<4xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: func @extract_contract1_int
// CHECK-SAME: %[[A:.*0]]: vector<4xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<4xi32>,
// CHECK-SAME: %[[C:.*2]]: i32
// CHECK:      %[[F:.*]] = arith.muli %[[A]], %[[B]] : vector<4xi32>
// CHECK:      %[[R:.*]] = vector.reduction <add>, %[[F]], %[[C]] : vector<4xi32> into i32
// CHECK:      return %[[R]] : i32

func.func @extract_contract1_int(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: i32) -> i32 {
  %0 = vector.contract #dotp_trait %arg0, %arg1, %arg2
    : vector<4xi32>, vector<4xi32> into i32
  return %0 : i32
}

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

// CHECK-LABEL: func @extract_contract2
// CHECK-SAME: %[[A:.*0]]: vector<2x3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK:      %[[R:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T0]], %[[B]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]] : vector<3xf32> into f32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : f32 into vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<2x3xf32>
// CHECK:      %[[T7:.*]] = arith.mulf %[[T5]], %[[B]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]] : vector<3xf32> into f32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : f32 into vector<2xf32>
// CHECK:      %[[T10:.*]] = arith.addf %[[T9]], %[[C]] : vector<2xf32>
// CHECK:      return %[[T10]] : vector<2xf32>

func.func @extract_contract2(%arg0: vector<2x3xf32>,
                        %arg1: vector<3xf32>,
                        %arg2: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<3xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @extract_contract2_int
// CHECK-SAME: %[[A:.*0]]: vector<2x3xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xi32>,
// CHECK-SAME: %[[C:.*2]]: vector<2xi32>
// CHECK:      %[[R:.*]] = arith.constant dense<0> : vector<2xi32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x3xi32>
// CHECK:      %[[T2:.*]] = arith.muli %[[T0]], %[[B]] : vector<3xi32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]] : vector<3xi32> into i32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : i32 into vector<2xi32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<2x3xi32>
// CHECK:      %[[T7:.*]] = arith.muli %[[T5]], %[[B]] : vector<3xi32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]] : vector<3xi32> into i32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : i32 into vector<2xi32>
// CHECK:      %[[T10:.*]] = arith.addi %[[T9]], %[[C]] : vector<2xi32>
// CHECK:      return %[[T10]] : vector<2xi32>
func.func @extract_contract2_int(%arg0: vector<2x3xi32>,
                        %arg1: vector<3xi32>,
                        %arg2: vector<2xi32>) -> vector<2xi32> {
  %0 = vector.contract #matvec_trait %arg0, %arg1, %arg2
    : vector<2x3xi32>, vector<3xi32> into vector<2xi32>
  return %0 : vector<2xi32>
}

#vecmat_accesses = [
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i)>
]
#vecmat_trait = {
  indexing_maps = #vecmat_accesses,
  iterator_types = ["parallel", "reduction"]
}

// CHECK-LABEL: func @extract_contract3
// CHECK-SAME: %[[A:.*0]]: vector<3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<2x3xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK:      %[[R:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[B]][0] : vector<2x3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T0]], %[[A]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]] : vector<3xf32> into f32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : f32 into vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[B]][1] : vector<2x3xf32>
// CHECK:      %[[T7:.*]] = arith.mulf %[[T5]], %[[A]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]] : vector<3xf32> into f32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : f32 into vector<2xf32>
// CHECK:      %[[T10:.*]] = arith.addf %[[T9]], %[[C]] : vector<2xf32>
// CHECK:      return %[[T10]] : vector<2xf32>

func.func @extract_contract3(%arg0: vector<3xf32>,
                        %arg1: vector<2x3xf32>,
                        %arg2: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #vecmat_trait %arg0, %arg1, %arg2
    : vector<3xf32>, vector<2x3xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @extract_contract4
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<2x2xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2x2xf32>
// CHECK:    %[[R:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK:    %[[Bt:.*]] = vector.transpose %arg1, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK:    %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x2xf32>
// CHECK:    %[[T2:.*]] = vector.extract %[[Bt]][0] : vector<2x2xf32>
// CHECK:    %[[T9:.*]] = arith.mulf %[[T0]], %[[T2]] : vector<2xf32>
// CHECK:    %[[T10:.*]] = vector.reduction <add>, %[[T9]] : vector<2xf32> into f32
// CHECK:    %[[T11:.*]] = vector.insert %[[T10]], %[[R]] [0, 0] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T12:.*]] = vector.extract %[[Bt]][1] : vector<2x2xf32>
// CHECK:    %[[T19:.*]] = arith.mulf %[[T0]], %[[T12]] : vector<2xf32>
// CHECK:    %[[T20:.*]] = vector.reduction <add>, %[[T19]] : vector<2xf32> into f32
// CHECK:    %[[T21:.*]] = vector.insert %[[T20]], %[[T11]] [0, 1] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T23:.*]] = vector.extract %[[A]][1] : vector<2x2xf32>
// CHECK:    %[[T24:.*]] = vector.extract %[[Bt]][0] : vector<2x2xf32>
// CHECK:    %[[T32:.*]] = arith.mulf %[[T23]], %[[T24]] : vector<2xf32>
// CHECK:    %[[T33:.*]] = vector.reduction <add>, %[[T32]] : vector<2xf32> into f32
// CHECK:    %[[T34:.*]] = vector.insert %[[T33]], %[[T21]] [1, 0] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T40:.*]] = vector.extract %[[Bt]][1] : vector<2x2xf32>
// CHECK:    %[[T41:.*]] = arith.mulf %[[T23]], %[[T40]] : vector<2xf32>
// CHECK:    %[[T42:.*]] = vector.reduction <add>, %[[T41]] : vector<2xf32> into f32
// CHECK:    %[[T43:.*]] = vector.insert %[[T42]], %[[T34]] [1, 1] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T52:.*]] = arith.addf %[[T43]], %[[C]] : vector<2x2xf32>
// CHECK:    return %[[T52]] : vector<2x2xf32>

func.func @extract_contract4(%arg0: vector<2x2xf32>,
                        %arg1: vector<2x2xf32>,
                        %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

#contraction2d_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> ()>
]
#contraction2d_trait = {
  indexing_maps = #contraction2d_accesses,
  iterator_types = ["reduction", "reduction"]
}

// CHECK-LABEL: func @full_contract1
// CHECK-SAME: %[[A:.*0]]: vector<2x3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<2x3xf32>,
// CHECK-SAME: %[[C:.*2]]: f32
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[B]][0] : vector<2x3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T0]], %[[T1]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]], %[[C]] : vector<3xf32> into f32
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<2x3xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[B]][1] : vector<2x3xf32>
// CHECK:      %[[T7:.*]] = arith.mulf %[[T5]], %[[T6]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]], %[[T3]] : vector<3xf32> into f32
// CHECK:      return %[[T8]] : f32

func.func @full_contract1(%arg0: vector<2x3xf32>,
                     %arg1: vector<2x3xf32>,
                     %arg2: f32) -> f32 {
  %0 = vector.contract #contraction2d_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<2x3xf32> into f32
  return %0 : f32
}

#contraction2d_trans_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> ()>
]
#contraction2d_trans_trait = {
  indexing_maps = #contraction2d_trans_accesses,
  iterator_types = ["reduction", "reduction"]
}

// CHECK-LABEL: func @full_contract2
// CHECK-SAME: %[[A:.*0]]: vector<2x3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3x2xf32>,
// CHECK-SAME: %[[C:.*2]]: f32
// CHECK:      %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[B]][0, 0] : vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T1]], %[[Z]] [0] : f32 into vector<3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[B]][1, 0] : vector<3x2xf32>
// CHECK:      %[[T6:.*]] = vector.insert %[[T4]], %[[T3]] [1] : f32 into vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.extract %[[B]][2, 0] : vector<3x2xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T7]], %[[T6]] [2] : f32 into vector<3xf32>
// CHECK:      %[[T10:.*]] = arith.mulf %[[T0]], %[[T9]] : vector<3xf32>
// CHECK:      %[[T11:.*]] = vector.reduction <add>, %[[T10]], %[[C]] : vector<3xf32> into f32
//
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][1] : vector<2x3xf32>
// CHECK:      %[[T13:.*]] = vector.extract %[[B]][0, 1] : vector<3x2xf
// CHECK:      %[[T15:.*]] = vector.insert %[[T13]], %[[Z]] [0] : f32 into vector<3xf32>
// CHECK:      %[[T16:.*]] = vector.extract %[[B]][1, 1] : vector<3x2xf32>
// CHECK:      %[[T18:.*]] = vector.insert %[[T16]], %[[T15]] [1] : f32 into vector<3xf32>
// CHECK:      %[[T19:.*]] = vector.extract %[[B]][2, 1] : vector<3x2xf32>
// CHECK:      %[[T21:.*]] = vector.insert %[[T19]], %[[T18]] [2] : f32 into vector<3xf32>
// CHECK:      %[[T22:.*]] = arith.mulf %[[T12]], %[[T21]] : vector<3xf32>
// CHECK:      %[[T23:.*]] = vector.reduction <add>, %[[T22]], %[[T11]] : vector<3xf32> into f32
// CHECK:      return %[[T23]] : f32

func.func @full_contract2(%arg0: vector<2x3xf32>,
                     %arg1: vector<3x2xf32>,
                     %arg2: f32) -> f32 {
  %0 = vector.contract #contraction2d_trans_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<3x2xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: func @outerproduct_noacc
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2xf32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T1]], %[[B]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C0]] [0] : vector<3xf32> into vector<2x3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1] : vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.splat %[[T4]] : vector<3xf32>
// CHECK:      %[[T6:.*]] = arith.mulf %[[T5]], %[[B]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xf32> into vector<2x3xf32>
// CHECK:      return %[[T7]] : vector<2x3xf32>

func.func @outerproduct_noacc(%arg0: vector<2xf32>,
                         %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %0 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %0: vector<2x3xf32>
}

// CHECK-LABEL: func @outerproduct_acc
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2x3xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2xf32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T2:.*]] = vector.extract %[[C]][0] : vector<2x3xf32>
// CHECK:      %[[T3:.*]] = vector.fma %[[T1]], %[[B]], %[[T2]] : vector<3xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[C0]] [0] : vector<3xf32> into vector<2x3xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<2xf32>
// CHECK:      %[[T6:.*]] = vector.splat %[[T5]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.extract %[[C]][1] : vector<2x3xf32>
// CHECK:      %[[T8:.*]] = vector.fma %[[T6]], %[[B]], %[[T7]] : vector<3xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : vector<3xf32> into vector<2x3xf32>
// CHECK:      return %[[T9]] : vector<2x3xf32>

func.func @outerproduct_acc(%arg0: vector<2xf32>,
                       %arg1: vector<3xf32>,
                       %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %0: vector<2x3xf32>
}

// CHECK-LABEL: func @outerproduct_noacc_int
// CHECK-SAME: %[[A:.*0]]: vector<2xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xi32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0> : vector<2x3xi32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2xi32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xi32>
// CHECK:      %[[T2:.*]] = arith.muli %[[T1]], %[[B]] : vector<3xi32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C0]] [0] : vector<3xi32> into vector<2x3xi32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1] : vector<2xi32>
// CHECK:      %[[T5:.*]] = vector.splat %[[T4]] : vector<3xi32>
// CHECK:      %[[T6:.*]] = arith.muli %[[T5]], %[[B]] : vector<3xi32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xi32> into vector<2x3xi32>
// CHECK:      return %[[T7]] : vector<2x3xi32>
func.func @outerproduct_noacc_int(%arg0: vector<2xi32>,
                             %arg1: vector<3xi32>) -> vector<2x3xi32> {
  %0 = vector.outerproduct %arg0, %arg1 : vector<2xi32>, vector<3xi32>
  return %0: vector<2x3xi32>
}

// CHECK-LABEL: func @outerproduct_acc_int
// CHECK-SAME: %[[A:.*0]]: vector<2xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xi32>,
// CHECK-SAME: %[[C:.*2]]: vector<2x3xi32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0> : vector<2x3xi32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2xi32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<3xi32>
// CHECK:      %[[T2:.*]] = vector.extract %[[C]][0] : vector<2x3xi32>
// CHECK:      %[[T3:.*]] = arith.muli %[[T1]], %[[B]] : vector<3xi32>
// CHECK:      %[[T4:.*]] = arith.addi %[[T3]], %[[T2]] : vector<3xi32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[C0]] [0] : vector<3xi32> into vector<2x3xi32>
// CHECK:      %[[T6:.*]] = vector.extract %[[A]][1] : vector<2xi32>
// CHECK:      %[[T7:.*]] = vector.splat %[[T6]] : vector<3xi32>
// CHECK:      %[[T8:.*]] = vector.extract %[[C]][1] : vector<2x3xi32>
// CHECK:      %[[T9:.*]] = arith.muli %[[T7]], %[[B]] : vector<3xi32>
// CHECK:      %[[T10:.*]] = arith.addi %[[T9]], %[[T8]] : vector<3xi32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T5]] [1] : vector<3xi32> into vector<2x3xi32>
// CHECK:      return %[[T11]] : vector<2x3xi32>
func.func @outerproduct_acc_int(%arg0: vector<2xi32>,
                           %arg1: vector<3xi32>,
                           %arg2: vector<2x3xi32>) -> vector<2x3xi32> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xi32>, vector<3xi32>
  return %0: vector<2x3xi32>
}

// CHECK-LABEL: func @axpy_fp(
// CHECK-SAME: %[[A:.*0]]: vector<16xf32>,
// CHECK-SAME: %[[B:.*1]]: f32)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xf32>
// CHECK: %[[T1:.*]] = arith.mulf %[[A]], %[[T0]] : vector<16xf32>
// CHECK: return %[[T1]] : vector<16xf32>
func.func @axpy_fp(%arg0: vector<16xf32>, %arg1: f32) -> vector<16xf32> {
   %0 = vector.outerproduct %arg0, %arg1: vector<16xf32>, f32
   return %0: vector<16xf32>
}

// CHECK-LABEL: func @axpy_fp_add(
// CHECK-SAME: %[[A:.*0]]: vector<16xf32>,
// CHECK-SAME: %[[B:.*1]]: f32,
// CHECK-SAME: %[[C:.*2]]: vector<16xf32>)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xf32>
// CHECK: %[[T1:.*]] = vector.fma %[[A]], %[[T0]], %[[C]] : vector<16xf32>
// CHECK: return %[[T1]] : vector<16xf32>
func.func @axpy_fp_add(%arg0: vector<16xf32>, %arg1: f32, %arg2 : vector<16xf32>) -> vector<16xf32> {
   %0 = vector.outerproduct %arg0, %arg1, %arg2: vector<16xf32>, f32
   return %0: vector<16xf32>
}

// CHECK-LABEL: func @axpy_int(
// CHECK-SAME: %[[A:.*0]]: vector<16xi32>,
// CHECK-SAME: %[[B:.*1]]: i32)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xi32>
// CHECK: %[[T1:.*]] = arith.muli %[[A]], %[[T0]] : vector<16xi32>
// CHECK: return %[[T1]] : vector<16xi32>
func.func @axpy_int(%arg0: vector<16xi32>, %arg1: i32) -> vector<16xi32> {
   %0 = vector.outerproduct %arg0, %arg1: vector<16xi32>, i32
   return %0: vector<16xi32>
}

// CHECK-LABEL: func @axpy_int_add(
// CHECK-SAME: %[[A:.*0]]: vector<16xi32>,
// CHECK-SAME: %[[B:.*1]]: i32,
// CHECK-SAME: %[[C:.*2]]: vector<16xi32>)
// CHECK: %[[T0:.*]] = vector.splat %[[B]] : vector<16xi32>
// CHECK: %[[T1:.*]] = arith.muli %[[A]], %[[T0]] : vector<16xi32>
// CHECK: %[[T2:.*]] = arith.addi %[[T1]], %[[C]] : vector<16xi32>
// CHECK: return %[[T2]] : vector<16xi32>
func.func @axpy_int_add(%arg0: vector<16xi32>, %arg1: i32, %arg2: vector<16xi32>) -> vector<16xi32> {
   %0 = vector.outerproduct %arg0, %arg1, %arg2: vector<16xi32>, i32
   return %0: vector<16xi32>
}

// CHECK-LABEL: func @nop_shape_cast
// CHECK-SAME: %[[A:.*]]: vector<16xf32>
// CHECK:      return %[[A]] : vector<16xf32>

func.func @nop_shape_cast(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @cancel_shape_cast
// FIXME: PR49590
// HECK-SAME: %[[A:.*]]: vector<16xf32>
// HECK:      return %[[A]] : vector<16xf32>

func.func @cancel_shape_cast(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<4x4xf32>
  %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
  return %1 : vector<16xf32>
}

// Shape up and downcasts for 2-D vectors, for supporting conversion to
// llvm.matrix operations
// CHECK-LABEL: func @shape_casts
func.func @shape_casts(%a: vector<2x2xf32>) -> (vector<4xf32>, vector<2x2xf32>) {
  // CHECK-DAG: %[[cst22:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
  // CHECK: %[[ex0:.*]] = vector.extract %{{.*}}[0] : vector<2x2xf32>
  //
  // CHECK: %[[in0:.*]] = vector.insert_strided_slice %[[ex0]], %[[cst]]
  // CHECK-SAME: {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
  //
  // CHECK: %[[ex1:.*]] = vector.extract %{{.*}}[1] : vector<2x2xf32>
  //
  // CHECK: %[[in2:.*]] = vector.insert_strided_slice %[[ex1]], %[[in0]]
  // CHECK-SAME: {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
  //
  %0 = vector.shape_cast %a : vector<2x2xf32> to vector<4xf32>
  // CHECK: %[[add:.*]] = arith.addf %[[in2]], %[[in2]] : vector<4xf32>
  %r0 = arith.addf %0, %0: vector<4xf32>
  //
  // CHECK: %[[ss0:.*]] = vector.extract_strided_slice %[[add]]
  // CHECK-SAME: {offsets = [0], sizes = [2], strides = [1]} :
  // CHECK-SAME: vector<4xf32> to vector<2xf32>
  //
  // CHECK: %[[res0:.*]] = vector.insert %[[ss0]], %[[cst22]] [0] :
  // CHECK-SAME: vector<2xf32> into vector<2x2xf32>
  //
  // CHECK: %[[s2:.*]] = vector.extract_strided_slice %[[add]]
  // CHECK-SAME: {offsets = [2], sizes = [2], strides = [1]} :
  // CHECK-SAME: vector<4xf32> to vector<2xf32>
  //
  // CHECK: %[[res1:.*]] = vector.insert %[[s2]], %[[res0]] [1] :
  // CHECK-SAME: vector<2xf32> into vector<2x2xf32>
  //
  %1 = vector.shape_cast %r0  : vector<4xf32> to vector<2x2xf32>
  // CHECK: return %[[add]], %[[res1]] : vector<4xf32>, vector<2x2xf32>
  return %r0, %1 : vector<4xf32>, vector<2x2xf32>
}

// CHECK-LABEL: func @shape_cast_2d2d
// CHECK-SAME: %[[A:.*]]: vector<3x2xf32>
// CHECK: %[[C:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK: %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<3x2xf32>
// CHECK: %[[T1:.*]] = vector.insert %[[T0]], %[[C]] [0, 0] : f32 into vector<2x3xf32>
// CHECK: %[[T2:.*]] = vector.extract %[[A]][0, 1] : vector<3x2xf32>
// CHECK: %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [0, 1] : f32 into vector<2x3xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[A]][1, 0] : vector<3x2xf32>
// CHECK: %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [0, 2] : f32 into vector<2x3xf32>
// CHECK: %[[T6:.*]] = vector.extract %[[A]][1, 1] : vector<3x2xf32>
// CHECK: %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [1, 0] : f32 into vector<2x3xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[A]][2, 0] : vector<3x2xf32>
// CHECK: %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [1, 1] : f32 into vector<2x3xf32>
// CHECK: %[[T10:.*]] = vector.extract %[[A]][2, 1] : vector<3x2xf32>
// CHECK: %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [1, 2] : f32 into vector<2x3xf32>
// CHECK: return %[[T11]] : vector<2x3xf32>

func.func @shape_cast_2d2d(%arg0 : vector<3x2xf32>) -> vector<2x3xf32> {
  %s = vector.shape_cast %arg0: vector<3x2xf32> to vector<2x3xf32>
  return %s : vector<2x3xf32>
}

// CHECK-LABEL: func @shape_cast_3d1d
// CHECK-SAME: %[[A:.*]]: vector<1x3x2xf32>
// CHECK: %[[C:.*]] = arith.constant dense<0.000000e+00> : vector<6xf32>
// CHECK: %[[T0:.*]] = vector.extract %[[A]][0, 0, 0] : vector<1x3x2xf32>
// CHECK: %[[T1:.*]] = vector.insert %[[T0]], %[[C]] [0] : f32 into vector<6xf32>
// CHECK: %[[T2:.*]] = vector.extract %[[A]][0, 0, 1] : vector<1x3x2xf32>
// CHECK: %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [1] : f32 into vector<6xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[A]][0, 1, 0] : vector<1x3x2xf32>
// CHECK: %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [2] : f32 into vector<6xf32>
// CHECK: %[[T6:.*]] = vector.extract %[[A]][0, 1, 1] : vector<1x3x2xf32>
// CHECK: %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [3] : f32 into vector<6xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[A]][0, 2, 0] : vector<1x3x2xf32>
// CHECK: %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [4] : f32 into vector<6xf32>
// CHECK: %[[T10:.*]] = vector.extract %[[A]][0, 2, 1] : vector<1x3x2xf32>
// CHECK: %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [5] : f32 into vector<6xf32>
// CHECK: return %[[T11]] : vector<6xf32>

func.func @shape_cast_3d1d(%arg0 : vector<1x3x2xf32>) -> vector<6xf32> {
  %s = vector.shape_cast %arg0 : vector<1x3x2xf32> to vector<6xf32>
  return %s : vector<6xf32>
}

// CHECK-LABEL: func @shape_cast_1d3d
// CHECK-SAME: %[[A:.*]]: vector<6xf32>
// CHECK: %[[C:.*]] = arith.constant dense<0.000000e+00> : vector<2x1x3xf32>
// CHECK: %[[T0:.*]] = vector.extract %[[A]][0] : vector<6xf32>
// CHECK: %[[T1:.*]] = vector.insert %[[T0]], %[[C]] [0, 0, 0] : f32 into vector<2x1x3xf32>
// CHECK: %[[T2:.*]] = vector.extract %[[A]][1] : vector<6xf32>
// CHECK: %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [0, 0, 1] : f32 into vector<2x1x3xf32>
// CHECK: %[[T4:.*]] = vector.extract %[[A]][2] : vector<6xf32>
// CHECK: %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [0, 0, 2] : f32 into vector<2x1x3xf32>
// CHECK: %[[T6:.*]] = vector.extract %[[A]][3] : vector<6xf32>
// CHECK: %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [1, 0, 0] : f32 into vector<2x1x3xf32>
// CHECK: %[[T8:.*]] = vector.extract %[[A]][4] : vector<6xf32>
// CHECK: %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [1, 0, 1] : f32 into vector<2x1x3xf32>
// CHECK: %[[T10:.*]] = vector.extract %[[A]][5] : vector<6xf32>
// CHECK: %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [1, 0, 2] : f32 into vector<2x1x3xf32>
// CHECK: return %[[T11]] : vector<2x1x3xf32>

func.func @shape_cast_1d3d(%arg0 : vector<6xf32>) -> vector<2x1x3xf32> {
  %s = vector.shape_cast %arg0 : vector<6xf32> to vector<2x1x3xf32>
  return %s : vector<2x1x3xf32>
}

// MATRIX-LABEL: func @matmul
// MATRIX-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// MATRIX-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x3xf32>,
// MATRIX-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      MATRIX:  %[[vcst:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
//      MATRIX:  %[[vcst_0:.*]] = arith.constant dense<0.000000e+00> : vector<12xf32>
//      MATRIX:  %[[vcst_1:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
//      MATRIX:  %[[a0:.*]] = vector.extract %[[A]][0] : vector<2x4xf32>
//      MATRIX:  %[[a1:.*]] = vector.insert_strided_slice %[[a0]], %[[vcst]] {offsets = [0], strides = [1]} : vector<4xf32> into vector<8xf32>
//      MATRIX:  %[[a2:.*]] = vector.extract %[[A]][1] : vector<2x4xf32>
//      MATRIX:  %[[a3:.*]] = vector.insert_strided_slice %[[a2]], %[[a1]] {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
//      MATRIX:  %[[b0:.*]] = vector.extract %[[B]][0] : vector<4x3xf32>
//      MATRIX:  %[[b1:.*]] = vector.insert_strided_slice %[[b0]], %[[vcst_0]] {offsets = [0], strides = [1]} : vector<3xf32> into vector<12xf32>
//      MATRIX:  %[[b2:.*]] = vector.extract %[[B]][1] : vector<4x3xf32>
//      MATRIX:  %[[b3:.*]] = vector.insert_strided_slice %[[b2]], %[[b1]] {offsets = [3], strides = [1]} : vector<3xf32> into vector<12xf32>
//      MATRIX:  %[[b4:.*]] = vector.extract %[[B]][2] : vector<4x3xf32>
//      MATRIX:  %[[b5:.*]] = vector.insert_strided_slice %[[b4]], %[[b3]] {offsets = [6], strides = [1]} : vector<3xf32> into vector<12xf32>
//      MATRIX:  %[[b6:.*]] = vector.extract %[[B]][3] : vector<4x3xf32>
//      MATRIX:  %[[b7:.*]] = vector.insert_strided_slice %[[b6]], %[[b5]] {offsets = [9], strides = [1]} : vector<3xf32> into vector<12xf32>
//      MATRIX:  %[[mm1:.*]] = vector.matrix_multiply %[[a3]], %[[b7]] {lhs_columns = 4 : i32, lhs_rows = 2 : i32, rhs_columns = 3 : i32} : (vector<8xf32>, vector<12xf32>) -> vector<6xf32>
//      MATRIX:  %[[mm2:.*]] = vector.extract_strided_slice %[[mm1]] {offsets = [0], sizes = [3], strides = [1]} : vector<6xf32> to vector<3xf32>
//      MATRIX:  %[[mm3:.*]] = vector.insert %[[mm2]], %[[vcst_1]] [0] : vector<3xf32> into vector<2x3xf32>
//      MATRIX:  %[[mm4:.*]] = vector.extract_strided_slice %[[mm1]] {offsets = [3], sizes = [3], strides = [1]} : vector<6xf32> to vector<3xf32>
//      MATRIX:  %[[mm5:.*]] = vector.insert %[[mm4]], %[[mm3]] [1] : vector<3xf32> into vector<2x3xf32>
//      MATRIX:  %[[mm6:.*]] = arith.addf %[[C]], %[[mm5]] : vector<2x3xf32>

// OUTERPRODUCT-LABEL: func @matmul
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x3xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
// OUTERPRODUCT-SAME:  : vector<2x4xf32> to vector<4x2xf32>
//
//      OUTERPRODUCT: %[[a0:.*]] = vector.extract %[[At]][0] : vector<4x2xf32>
//      OUTERPRODUCT: %[[b0:.*]] = vector.extract %[[B]][0] : vector<4x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
// OUTERPRODUCT-SAME:  : vector<2xf32>, vector<3xf32>
//
//      OUTERPRODUCT: %[[a1:.*]] = vector.extract %[[At]][1] : vector<4x2xf32>
//      OUTERPRODUCT: %[[b1:.*]] = vector.extract %[[B]][1] : vector<4x3xf32>
//      OUTERPRODUCT: %[[c1:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[c0]]
// OUTERPRODUCT-SAME:  : vector<2xf32>, vector<3xf32>
//
//      OUTERPRODUCT: %[[a2:.*]] = vector.extract %[[At]][2] : vector<4x2xf32>
//      OUTERPRODUCT: %[[b2:.*]] = vector.extract %[[B]][2] : vector<4x3xf32>
//      OUTERPRODUCT: %[[c2:.*]] = vector.outerproduct %[[a2]], %[[b2]], %[[c1]]
// OUTERPRODUCT-SAME:  : vector<2xf32>, vector<3xf32>
//
//      OUTERPRODUCT: %[[a3:.*]] = vector.extract %[[At]][3] : vector<4x2xf32>
//      OUTERPRODUCT: %[[b3:.*]] = vector.extract %[[B]][3] : vector<4x3xf32>
//      OUTERPRODUCT: %[[c3:.*]] = vector.outerproduct %[[a3]], %[[b3]], %[[c2]]
// OUTERPRODUCT-SAME:  : vector<2xf32>, vector<3xf32>
//
//      OUTERPRODUCT: return %[[c3]] : vector<2x3xf32>

// REDUCE-LABEL: func @matmul
// REDUCE-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// REDUCE-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x3xf32>,
// REDUCE-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//
//      REDUCE: %[[RES:.*]] = arith.constant dense<0.000000e+00> : vector<2x3xf32>
//      REDUCE: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
// REDUCE-SAME:  : vector<4x3f32> to vector<3x4xf32>
//
//      REDUCE: %[[a0:.*]] = vector.extract %[[A]][0] : vector<2x4xf32>
// REDUCE-NEXT: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<3x4xf32>
// REDUCE-NEXT: %[[ab00:.*]] = mul %[[a0]], %[[b0]] : vector<4xf32>
// REDUCE-NEXT: %[[s00:.*]] = vector.reduction <add>, %[[ab00]] : vector<4xf32> into f32
// REDUCE-NEXT: %[[r00:.*]] = vector.insert %[[s00]], %[[RES]] [0, 0] : f32 into vector<2x3xf32>
//
//      ...
//
//      REDUCE: %[[a1:.*]] = vector.extract %[[A]][1] : vector<2x4xf32>
// REDUCE-NEXT: %[[b2:.*]] = vector.extract %[[Bt]][2] : vector<3x4xf32>
// REDUCE-NEXT: %[[ab12:.*]] = mul %[[a1]], %[[b02]] : vector<4xf32>
// REDUCE-NEXT: %[[s12:.*]] = vector.reduction <add>, %[[ab12]] : vector<4xf32> into f32
// REDUCE-NEXT: %[[r12:.*]] = vector.insert %[[s12]], %{{.*}} [1, 2] : f32 into vector<2x3xf32>
//
//      REDUCE: return %[[c3]] : vector<2x3xf32>
func.func @matmul(%arg0: vector<2x4xf32>,
                          %arg1: vector<4x3xf32>,
                          %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @broadcast_vec1d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = vector.splat %[[A]] : vector<2xf32>
// CHECK:      return %[[T0]] : vector<2xf32>

func.func @broadcast_vec1d_from_scalar(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @broadcast_vec2d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = vector.splat %[[A]] : vector<2x3xf32>
// CHECK:      return %[[T0]] : vector<2x3xf32>

func.func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = vector.splat %[[A]] : vector<2x3x4xf32>
// CHECK:      return %[[T0]] : vector<2x3x4xf32>

func.func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_vec1d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      return %[[A]] : vector<2xf32>

func.func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @broadcast_vec2d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[C0]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      return %[[T2]] : vector<3x2xf32>

func.func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[C1:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[C0]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C1]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T2]], %[[T3]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T2]], %[[T4]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T6:.*]] = vector.insert %[[T2]], %[[T5]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:       return %[[T6]] : vector<4x3x2xf32>

func.func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_vec2d
// CHECK-SAME: %[[A:.*0]]: vector<3x2xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[C0]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[A]], %[[T2]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      return %[[T3]] : vector<4x3x2xf32>

func.func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @broadcast_stretch
// CHECK-SAME: %[[A:.*0]]: vector<1xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<1xf32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<4xf32>
// CHECK:      return %[[T1]] : vector<4xf32>

func.func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @broadcast_stretch_at_start
// CHECK-SAME: %[[A:.*0]]: vector<1x4xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<3x4xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<1x4xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[C0]] [0] : vector<4xf32> into vector<3x4xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[T0]], %[[T1]] [1] : vector<4xf32> into vector<3x4xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T0]], %[[T2]] [2] : vector<4xf32> into vector<3x4xf32>
// CHECK:      return %[[T3]] : vector<3x4xf32>

func.func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// CHECK-LABEL: func @broadcast_stretch_at_end
// CHECK-SAME: %[[A:.*0]]: vector<4x1xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<4x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<4x1xf32>
// CHECK:      %[[T2:.*]] = vector.splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C0]] [0] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1, 0] : vector<4x1xf32>
// CHECK:      %[[T6:.*]] = vector.splat %[[T4]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T8:.*]] = vector.extract %[[A]][2, 0] : vector<4x1xf32>
// CHECK:      %[[T10:.*]] = vector.splat %[[T8]] : vector<3xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T7]] [2] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][3, 0] : vector<4x1xf32>
// CHECK:      %[[T14:.*]] = vector.splat %[[T12]] : vector<3xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T14]], %[[T11]] [3] : vector<3xf32> into vector<4x3xf32>
// CHECK:      return %[[T15]] : vector<4x3xf32>

func.func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}

// CHECK-LABEL: func @broadcast_stretch_in_middle
// CHECK-SAME: %[[A:.*0]]: vector<4x1x2xf32>
// CHECK:      %[[C0:.*]] = arith.constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:      %[[C1:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<4x1x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[T0]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T0]], %[[T2]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T0]], %[[T3]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[C0]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[A]][1, 0] : vector<4x1x2xf32>
// CHECK:      %[[T8:.*]] = vector.insert %[[T6]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T6]], %[[T8]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T10:.*]] = vector.insert %[[T6]], %[[T9]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T5]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][2, 0] : vector<4x1x2xf32>
// CHECK:      %[[T14:.*]] = vector.insert %[[T12]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T12]], %[[T14]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T16:.*]] = vector.insert %[[T12]], %[[T15]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T17:.*]] = vector.insert %[[T16]], %[[T11]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T18:.*]] = vector.extract %[[A]][3, 0] : vector<4x1x2xf32>
// CHECK:      %[[T20:.*]] = vector.insert %[[T18]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T21:.*]] = vector.insert %[[T18]], %[[T20]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T22:.*]] = vector.insert %[[T18]], %[[T21]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T23:.*]] = vector.insert %[[T22]], %[[T17]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      return %[[T23]] : vector<4x3x2xf32>

func.func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @genbool_1d
// CHECK: %[[T0:.*]] = arith.constant dense<[true, true, true, true, false, false, false, false]> : vector<8xi1>
// CHECK: return %[[T0]] : vector<8xi1>

func.func @genbool_1d() -> vector<8xi1> {
  %0 = vector.constant_mask [4] : vector<8xi1>
  return %0 : vector<8xi1>
}

// CHECK-LABEL: func @genbool_2d
// CHECK: %[[C1:.*]] = arith.constant dense<[true, true, false, false]> : vector<4xi1>
// CHECK: %[[C2:.*]] = arith.constant dense<false> : vector<4x4xi1>
// CHECK: %[[T0:.*]] = vector.insert %[[C1]], %[[C2]] [0] : vector<4xi1> into vector<4x4xi1>
// CHECK: %[[T1:.*]] = vector.insert %[[C1]], %[[T0]] [1] : vector<4xi1> into vector<4x4xi1>
// CHECK: return %[[T1]] : vector<4x4xi1>

func.func @genbool_2d() -> vector<4x4xi1> {
  %v = vector.constant_mask [2, 2] : vector<4x4xi1>
  return %v: vector<4x4xi1>
}

// CHECK-LABEL: func @genbool_3d
// CHECK: %[[C1:.*]] = arith.constant dense<[true, true, true, false]> : vector<4xi1>
// CHECK: %[[C2:.*]] = arith.constant dense<false> : vector<3x4xi1>
// CHECK: %[[C3:.*]] = arith.constant dense<false> : vector<2x3x4xi1>
// CHECK: %[[T0:.*]] = vector.insert %[[C1]], %[[C2]] [0] : vector<4xi1> into vector<3x4xi1>
// CHECK: %[[T1:.*]] = vector.insert %[[T0]], %[[C3]] [0] : vector<3x4xi1> into vector<2x3x4xi1>
// CHECK: return %[[T1]] : vector<2x3x4xi1>

func.func @genbool_3d() -> vector<2x3x4xi1> {
  %v = vector.constant_mask [1, 1, 3] : vector<2x3x4xi1>
  return %v: vector<2x3x4xi1>
}

// CHECK-LABEL: func @genbool_var_1d(
// CHECK-SAME: %[[A:.*]]: index)
// CHECK:      %[[T0:.*]] = vector.create_mask %[[A]] : vector<3xi1>
// CHECK:      return %[[T0]] : vector<3xi1>

func.func @genbool_var_1d(%arg0: index) -> vector<3xi1> {
  %0 = vector.create_mask %arg0 : vector<3xi1>
  return %0 : vector<3xi1>
}

// CHECK-LABEL: func @genbool_var_2d(
// CHECK-SAME: %[[A:.*0]]: index,
// CHECK-SAME: %[[B:.*1]]: index)
// CHECK:      %[[C1:.*]] = arith.constant dense<false> : vector<3xi1>
// CHECK:      %[[C2:.*]] = arith.constant dense<false> : vector<2x3xi1>
// CHECK:      %[[c0:.*]] = arith.constant 0 : index
// CHECK:      %[[c1:.*]] = arith.constant 1 : index
// CHECK:      %[[T0:.*]] = vector.create_mask %[[B]] : vector<3xi1>
// CHECK:      %[[T1:.*]] = arith.cmpi slt, %[[c0]], %[[A]] : index
// CHECK:      %[[T2:.*]] = arith.select %[[T1]], %[[T0]], %[[C1]] : vector<3xi1>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C2]] [0] : vector<3xi1> into vector<2x3xi1>
// CHECK:      %[[T4:.*]] = arith.cmpi slt, %[[c1]], %[[A]] : index
// CHECK:      %[[T5:.*]] = arith.select %[[T4]], %[[T0]], %[[C1]] : vector<3xi1>
// CHECK:      %[[T6:.*]] = vector.insert %[[T5]], %[[T3]] [1] : vector<3xi1> into vector<2x3xi1>
// CHECK:      return %[[T6]] : vector<2x3xi1>

func.func @genbool_var_2d(%arg0: index, %arg1: index) -> vector<2x3xi1> {
  %0 = vector.create_mask %arg0, %arg1 : vector<2x3xi1>
  return %0 : vector<2x3xi1>
}

// CHECK-LABEL: func @genbool_var_3d(
// CHECK-SAME: %[[A:.*0]]: index,
// CHECK-SAME: %[[B:.*1]]: index,
// CHECK-SAME: %[[C:.*2]]: index)
// CHECK-DAG:  %[[C1:.*]] = arith.constant dense<false> : vector<7xi1>
// CHECK-DAG:  %[[C2:.*]] = arith.constant dense<false> : vector<1x7xi1>
// CHECK-DAG:  %[[C3:.*]] = arith.constant dense<false> : vector<2x1x7xi1>
// CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[c1:.*]] = arith.constant 1 : index
// CHECK:      %[[T0:.*]] = vector.create_mask %[[C]] : vector<7xi1>
// CHECK:      %[[T1:.*]] = arith.cmpi slt, %[[c0]], %[[B]] : index
// CHECK:      %[[T2:.*]] = arith.select %[[T1]], %[[T0]], %[[C1]] : vector<7xi1>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C2]] [0] : vector<7xi1> into vector<1x7xi1>
// CHECK:      %[[T4:.*]] = arith.cmpi slt, %[[c0]], %[[A]] : index
// CHECK:      %[[T5:.*]] = arith.select %[[T4]], %[[T3]], %[[C2]] : vector<1x7xi1>
// CHECK:      %[[T6:.*]] = vector.insert %[[T5]], %[[C3]] [0] : vector<1x7xi1> into vector<2x1x7xi1>
// CHECK:      %[[T7:.*]] = arith.cmpi slt, %[[c1]], %[[A]] : index
// CHECK:      %[[T8:.*]] = arith.select %[[T7]], %[[T3]], %[[C2]] : vector<1x7xi1>
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T6]] [1] : vector<1x7xi1> into vector<2x1x7xi1>
// CHECK:      return %[[T9]] : vector<2x1x7xi1>

func.func @genbool_var_3d(%arg0: index, %arg1: index, %arg2: index) -> vector<2x1x7xi1> {
  %0 = vector.create_mask %arg0, %arg1, %arg2 : vector<2x1x7xi1>
  return %0 : vector<2x1x7xi1>
}

#matmat_accesses_0 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_0 = {
  indexing_maps = #matmat_accesses_0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_0
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      OUTERPRODUCT: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      OUTERPRODUCT: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<2x3xf32>
func.func @matmul_0(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_0 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// OUTERPRODUCT-LABEL: func @matmul_0_mixed
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf16>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf16>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      OUTERPRODUCT: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf16>
//      OUTERPRODUCT: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf16>
//      OUTERPRODUCT: %[[a1:.*]] = arith.extf %[[a0]] : vector<2xf16> to vector<2xf32>
//      OUTERPRODUCT: %[[b1:.*]] = arith.extf %[[b0]] : vector<3xf16> to vector<3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<2x3xf32>
func.func @matmul_0_mixed(%arg0: vector<2x1xf16>, %arg1: vector<1x3xf16>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_0 %arg0, %arg1, %arg2
    : vector<2x1xf16>, vector<1x3xf16> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_1 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_1 = {
  indexing_maps = #matmat_accesses_1,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_1
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<3x1xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      OUTERPRODUCT: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      OUTERPRODUCT: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      OUTERPRODUCT: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<2x3xf32>
func.func @matmul_1(%arg0: vector<2x1xf32>, %arg1: vector<3x1xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_1 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<3x1xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_2 = [
  affine_map<(m, n, k) -> (k, m)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_2 = {
  indexing_maps = #matmat_accesses_2,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_2
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      OUTERPRODUCT: %[[a0:.*]] = vector.extract %[[A]][0] : vector<1x2xf32>
//      OUTERPRODUCT: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<2x3xf32>
func.func @matmul_2(%arg0: vector<1x2xf32>, %arg1: vector<1x3xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_2 %arg0, %arg1, %arg2
    : vector<1x2xf32>, vector<1x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_3 = [
  affine_map<(m, n, k) -> (k, m)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_3 = {
  indexing_maps = #matmat_accesses_3,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_3
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<3x1xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      OUTERPRODUCT: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      OUTERPRODUCT: %[[a0:.*]] = vector.extract %[[A]][0] : vector<1x2xf32>
//      OUTERPRODUCT: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<2x3xf32>
func.func @matmul_3(%arg0: vector<1x2xf32>, %arg1: vector<3x1xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_3 %arg0, %arg1, %arg2
    : vector<1x2xf32>, vector<3x1xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_4 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_4 = {
  indexing_maps = #matmat_accesses_4,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_4
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      OUTERPRODUCT: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<3x2xf32>
func.func @matmul_4(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_4 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

#matmat_accesses_5 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_5 = {
  indexing_maps = #matmat_accesses_5,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_5
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      OUTERPRODUCT-DAG: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      OUTERPRODUCT-DAG: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<3x2xf32>
func.func @matmul_5(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_5 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

#matmat_accesses_6 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_6 = {
  indexing_maps = #matmat_accesses_6,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_6
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      OUTERPRODUCT-DAG: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      OUTERPRODUCT-DAG: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<3x2xf32>
func.func @matmul_6(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_6 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

#matmat_accesses_7 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_7 = {
  indexing_maps = #matmat_accesses_7,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// OUTERPRODUCT-LABEL: func @matmul_7
// OUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// OUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// OUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      OUTERPRODUCT: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      OUTERPRODUCT-DAG: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      OUTERPRODUCT-DAG: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      OUTERPRODUCT: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      OUTERPRODUCT: return %[[c0]] : vector<3x2xf32>
func.func @matmul_7(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_7 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// FILTEROUTERPRODUCT-LABEL: func @matmul_4_filtered
// FILTEROUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<4x4xf32>,
// FILTEROUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x4xf32>,
// FILTEROUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<4x4xf32>
//      FILTEROUTERPRODUCT: %[[c0:.*]] = vector.contract {{{.*}}} %[[A]], %[[B]], %[[C]]
func.func @matmul_4_filtered(%arg0: vector<4x4xf32>, %arg1: vector<4x4xf32>, %arg2: vector<4x4xf32>)
-> vector<4x4xf32>
{
  %0 = vector.contract #matmat_trait_0 %arg0, %arg1, %arg2
    : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// FILTEROUTERPRODUCT-LABEL: func @matmul_4_not_filtered
// FILTEROUTERPRODUCT-SAME: %[[A:[a-zA-Z0-9]*]]: vector<3x4xf32>,
// FILTEROUTERPRODUCT-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x4xf32>,
// FILTEROUTERPRODUCT-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x4xf32>
//      FILTEROUTERPRODUCT: %[[c0:.*]] = vector.contract {{{.*}}} %[[A]], %[[B]], %[[C]]
func.func @matmul_4_not_filtered(%arg0: vector<3x4xf32>, %arg1: vector<4x4xf32>, %arg2: vector<3x4xf32>)
-> vector<3x4xf32>
{
  %0 = vector.contract #matmat_trait_0 %arg0, %arg1, %arg2
    : vector<3x4xf32>, vector<4x4xf32> into vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// PARALLEL-LABEL: func @parrallel_contract_lowering
//       PARALLEL:   %[[E0:.*]] = vector.extract %{{.*}}[0, 0] : vector<1x1x4xf32>
//       PARALLEL:   %[[E1:.*]] = vector.extract %{{.*}}[0, 0] : vector<1x1x4xf32>
//       PARALLEL:   %[[F:.*]] = vector.fma %[[E0]], %[[E1]], %{{.*}} : vector<4xf32>
//       PARALLEL:   return %[[F]] : vector<4xf32>
func.func @parrallel_contract_lowering(%arg0: vector<1x1x4xf32>, %arg1: vector<1x1x4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<1x1x4xf32>, vector<1x1x4xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// PARALLEL-LABEL: func @parrallel_contract_lowering_broadcast
//       PARALLEL:   %[[B:.*]] = vector.broadcast %{{.*}} : vector<1x1xf32> to vector<4x1x1xf32>
//       PARALLEL:   %[[T:.*]] = vector.transpose %[[B]], [1, 2, 0] : vector<4x1x1xf32> to vector<1x1x4xf32>
//       PARALLEL:   %[[E0:.*]] = vector.extract %[[T]][0, 0] : vector<1x1x4xf32>
//       PARALLEL:   %[[E1:.*]] = vector.extract %{{.*}}[0, 0] : vector<1x1x4xf32>
//       PARALLEL:   %[[F:.*]] = vector.fma %[[E0]], %[[E1]], %{{.*}} : vector<4xf32>
//       PARALLEL:   return %[[F]] : vector<4xf32>
func.func @parrallel_contract_lowering_broadcast(%arg0: vector<1x1xf32>, %arg1: vector<1x1x4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<1x1xf32>, vector<1x1x4xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// PARALLEL-LABEL: func @parrallel_contract_lowering
//       PARALLEL:   %[[B:.*]] = vector.broadcast %{{.*}} : vector<1x1xf32> to vector<4x1x1xf32>
//       PARALLEL:   %[[T0:.*]] = vector.transpose %[[B]], [1, 2, 0] : vector<4x1x1xf32> to vector<1x1x4xf32>
//       PARALLEL:   %[[T1:.*]] = vector.transpose %{{.*}}, [0, 2, 1] : vector<1x4x1xf32> to vector<1x1x4xf32>
//       PARALLEL:   %[[E0:.*]] = vector.extract %[[T0]][0, 0] : vector<1x1x4xf32>
//       PARALLEL:   %[[E1:.*]] = vector.extract %[[T1]][0, 0] : vector<1x1x4xf32>
//       PARALLEL:   %[[F:.*]] = vector.fma %[[E0]], %[[E1]], %arg2 : vector<4xf32>
//       PARALLEL:   return %[[F]] : vector<4xf32>
func.func @parrallel_contract_lowering_transpose(%arg0: vector<1x1xf32>, %arg1: vector<1x4x1xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<1x1xf32>, vector<1x4x1xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// PARALLEL-LABEL: func @parrallel_contract_lowering_scalar
//       PARALLEL:   %[[E0:.*]] = vector.extract %{{.*}}[0, 0] : vector<1x1xf32>
//       PARALLEL:   %[[E1:.*]] = vector.extract %{{.*}}[0, 0] : vector<1x1xf32>
//       PARALLEL:   %[[M:.*]] = arith.mulf %[[E0]], %[[E1]] : f32
//       PARALLEL:   %[[A:.*]] = arith.addf %[[M]], %{{.*}} : f32
//       PARALLEL:   return %[[A]] : f32
func.func @parrallel_contract_lowering_scalar(%arg0: vector<1x1xf32>, %arg1: vector<1x1xf32>, %arg2: f32) -> f32 {
  %0 = vector.contract {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> ()>],
    iterator_types = ["reduction", "reduction"], kind = #vector.kind<add>}
  %arg0, %arg1, %arg2 : vector<1x1xf32>, vector<1x1xf32> into f32
  return %0 : f32
}
