// RUN: mlir-opt %s -test-vector-contraction-conversion | FileCheck %s
// RUN: mlir-opt %s -test-vector-contraction-conversion=vector-lower-matrix-intrinsics=1 | FileCheck %s --check-prefix=MATRIX

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
// CHECK:      %[[Z:.*]] = constant dense<0.000000e+00> : vector<4xf32>
// CHECK:      %[[F:.*]] = vector.fma %[[A]], %[[B]], %[[Z]] : vector<4xf32>
// CHECK:      %[[R:.*]] = vector.reduction "add", %[[F]], %[[C]] : vector<4xf32> into f32
// CHECK:      return %[[R]] : f32

func @extract_contract1(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: f32) -> f32 {
  %0 = vector.contract #dotp_trait %arg0, %arg1, %arg2
    : vector<4xf32>, vector<4xf32> into f32
  return %0 : f32
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
// CHECK:      %[[R:.*]] = constant dense<0.000000e+00> : vector<2xf32>
// CHECK:      %[[Z:.*]] = constant dense<0.000000e+00> : vector<3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[C]][0] : vector<2xf32>
// CHECK:      %[[T2:.*]] = vector.fma %[[T0]], %[[B]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction "add", %[[T2]], %[[T1]] : vector<3xf32> into f32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : f32 into vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<2x3xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[C]][1] : vector<2xf32>
// CHECK:      %[[T7:.*]] = vector.fma %[[T5]], %[[B]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction "add", %[[T7]], %[[T6]] : vector<3xf32> into f32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : f32 into vector<2xf32>
// CHECK:      return %[[T9]] : vector<2xf32>

func @extract_contract2(%arg0: vector<2x3xf32>,
                        %arg1: vector<3xf32>,
			%arg2: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<3xf32> into vector<2xf32>
  return %0 : vector<2xf32>
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
// CHECK:      %[[R:.*]] = constant dense<0.000000e+00> : vector<2xf32>
// CHECK:      %[[Z:.*]] = constant dense<0.000000e+00> : vector<3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[B]][0] : vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[C]][0] : vector<2xf32>
// CHECK:      %[[T2:.*]] = vector.fma %[[A]], %[[T0]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction "add", %[[T2]], %[[T1]] : vector<3xf32> into f32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : f32 into vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[B]][1] : vector<2x3xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[C]][1] : vector<2xf32>
// CHECK:      %[[T7:.*]] = vector.fma %[[A]], %[[T5]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction "add", %[[T7]], %[[T6]] : vector<3xf32> into f32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : f32 into vector<2xf32>
// CHECK:      return %[[T9]] : vector<2xf32>

func @extract_contract3(%arg0: vector<3xf32>,
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
// CHECK:    %[[R:.*]] = constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK:    %[[Z:.*]] = constant dense<0.000000e+00> : vector<2xf32>
// CHECK:    %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x2xf32>
// CHECK:    %[[T1:.*]] = vector.extract %[[C]][0] : vector<2x2xf32>
// CHECK:    %[[T2:.*]] = vector.extract %[[B]][0] : vector<2x2xf32>
// CHECK:    %[[T3:.*]] = vector.extract %[[T2]][0] : vector<2xf32>
// CHECK:    %[[T4:.*]] = vector.insert %[[T3]], %[[Z]] [0] : f32 into vector<2xf32>
// CHECK:    %[[T5:.*]] = vector.extract %[[B]][1] : vector<2x2xf32>
// CHECK:    %[[T6:.*]] = vector.extract %[[T5]][0] : vector<2xf32>
// CHECK:    %[[T7:.*]] = vector.insert %[[T6]], %[[T4]] [1] : f32 into vector<2xf32>
// CHECK:    %[[T8:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK:    %[[T9:.*]] = vector.fma %[[T0]], %[[T7]], %[[Z]] : vector<2xf32>
// CHECK:    %[[T10:.*]] = vector.reduction "add", %[[T9]], %[[T8]] : vector<2xf32> into f32
// CHECK:    %[[T11:.*]] = vector.insert %[[T10]], %[[Z]] [0] : f32 into vector<2xf32>
// CHECK:    %[[T12:.*]] = vector.extract %[[B]][0] : vector<2x2xf32>
// CHECK:    %[[T13:.*]] = vector.extract %[[T12]][1] : vector<2xf32>
// CHECK:    %[[T14:.*]] = vector.insert %[[T13]], %[[Z]] [0] : f32 into vector<2xf32>
// CHECK:    %[[T15:.*]] = vector.extract %[[B]][1] : vector<2x2xf32>
// CHECK:    %[[T16:.*]] = vector.extract %[[T15]][1] : vector<2xf32>
// CHECK:    %[[T17:.*]] = vector.insert %[[T16]], %[[T14]] [1] : f32 into vector<2xf32>
// CHECK:    %[[T18:.*]] = vector.extract %[[T1]][1] : vector<2xf32>
// CHECK:    %[[T19:.*]] = vector.fma %[[T0]], %[[T17]], %[[Z]] : vector<2xf32>
// CHECK:    %[[T20:.*]] = vector.reduction "add", %[[T19]], %[[T18]] : vector<2xf32> into f32
// CHECK:    %[[T21:.*]] = vector.insert %[[T20]], %[[T11]] [1] : f32 into vector<2xf32>
// CHECK:    %[[T22:.*]] = vector.insert %[[T21]], %[[R]] [0] : vector<2xf32> into vector<2x2xf32>
// CHECK:    %[[T23:.*]] = vector.extract %[[A]][1] : vector<2x2xf32>
// CHECK:    %[[T24:.*]] = vector.extract %[[C]][1] : vector<2x2xf32>
// CHECK:    %[[T25:.*]] = vector.extract %[[B]][0] : vector<2x2xf32>
// CHECK:    %[[T26:.*]] = vector.extract %[[T25]][0] : vector<2xf32>
// CHECK:    %[[T27:.*]] = vector.insert %[[T26]], %[[Z]] [0] : f32 into vector<2xf32>
// CHECK:    %[[T28:.*]] = vector.extract %[[B]][1] : vector<2x2xf32>
// CHECK:    %[[T29:.*]] = vector.extract %[[T28]][0] : vector<2xf32>
// CHECK:    %[[T30:.*]] = vector.insert %[[T29]], %[[T27]] [1] : f32 into vector<2xf32>
// CHECK:    %[[T31:.*]] = vector.extract %[[T24]][0] : vector<2xf32>
// CHECK:    %[[T32:.*]] = vector.fma %[[T23]], %[[T30]], %[[Z]] : vector<2xf32>
// CHECK:    %[[T33:.*]] = vector.reduction "add", %[[T32]], %[[T31]] : vector<2xf32> into f32
// CHECK:    %[[T34:.*]] = vector.insert %[[T33]], %[[Z]] [0] : f32 into vector<2xf32>
// CHECK:    %[[T35:.*]] = vector.extract %[[B]][0] : vector<2x2xf32>
// CHECK:    %[[T36:.*]] = vector.extract %[[T35]][1] : vector<2xf32>
// CHECK:    %[[T37:.*]] = vector.insert %[[T36]], %[[Z]] [0] : f32 into vector<2xf32>
// CHECK:    %[[T38:.*]] = vector.extract %[[B]][1] : vector<2x2xf32>
// CHECK:    %[[T39:.*]] = vector.extract %[[T38]][1] : vector<2xf32>
// CHECK:    %[[T40:.*]] = vector.insert %[[T39]], %[[T37]] [1] : f32 into vector<2xf32>
// CHECK:    %[[T41:.*]] = vector.extract %[[T24]][1] : vector<2xf32>
// CHECK:    %[[T42:.*]] = vector.fma %[[T23]], %[[T40]], %[[Z]] : vector<2xf32>
// CHECK:    %[[T43:.*]] = vector.reduction "add", %[[T42]], %[[T41]] : vector<2xf32> into f32
// CHECK:    %[[T44:.*]] = vector.insert %[[T43]], %[[T34]] [1] : f32 into vector<2xf32>
// CHECK:    %[[T45:.*]] = vector.insert %[[T44]], %[[T22]] [1] : vector<2xf32> into vector<2x2xf32>
// CHECK:    return %[[T45]] : vector<2x2xf32>

func @extract_contract4(%arg0: vector<2x2xf32>,
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
// CHECK:      %[[Z:.*]] = constant dense<0.000000e+00> : vector<3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[B]][0] : vector<2x3xf32>
// CHECK:      %[[T2:.*]] = vector.fma %[[T0]], %[[T1]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction "add", %[[T2]], %[[C]] : vector<3xf32> into f32
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1] : vector<2x3xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[B]][1] : vector<2x3xf32>
// CHECK:      %[[T6:.*]] = vector.fma %[[T4]], %[[T5]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.reduction "add", %[[T6]], %[[T3]] : vector<3xf32> into f32
// CHECK:      return %[[T7]] : f32

func @full_contract1(%arg0: vector<2x3xf32>,
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
// CHECK:      %[[Z:.*]] = constant dense<0.000000e+00> : vector<3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[B]][0] : vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.extract %[[T1]][0] : vector<2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[Z]] [0] : f32 into vector<3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[B]][1] : vector<3x2xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[T4]][0] : vector<2xf32>
// CHECK:      %[[T6:.*]] = vector.insert %[[T5]], %[[T3]] [1] : f32 into vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.extract %[[B]][2] : vector<3x2xf32>
// CHECK:      %[[T8:.*]] = vector.extract %[[T7]][0] : vector<2xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T6]] [2] : f32 into vector<3xf32>
// CHECK:      %[[T10:.*]] = vector.fma %[[T0]], %[[T9]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T11:.*]] = vector.reduction "add", %[[T10]], %[[C]] : vector<3xf32> into f32
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][1] : vector<2x3xf32>
// CHECK:      %[[T13:.*]] = vector.extract %[[B]][0] : vector<3x2xf32>
// CHECK:      %[[T14:.*]] = vector.extract %[[T13]][1] : vector<2xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T14]], %[[Z]] [0] : f32 into vector<3xf32>
// CHECK:      %[[T16:.*]] = vector.extract %[[B]][1] : vector<3x2xf32>
// CHECK:      %[[T17:.*]] = vector.extract %[[T16]][1] : vector<2xf32>
// CHECK:      %[[T18:.*]] = vector.insert %[[T17]], %[[T15]] [1] : f32 into vector<3xf32>
// CHECK:      %[[T19:.*]] = vector.extract %[[B]][2] : vector<3x2xf32>
// CHECK:      %[[T20:.*]] = vector.extract %[[T19]][1] : vector<2xf32>
// CHECK:      %[[T21:.*]] = vector.insert %[[T20]], %[[T18]] [2] : f32 into vector<3xf32>
// CHECK:      %[[T22:.*]] = vector.fma %[[T12]], %[[T21]], %[[Z]] : vector<3xf32>
// CHECK:      %[[T23:.*]] = vector.reduction "add", %[[T22]], %[[T11]] : vector<3xf32> into f32
// CHECK:      return %[[T23]] : f32

func @full_contract2(%arg0: vector<2x3xf32>,
                     %arg1: vector<3x2xf32>,
		     %arg2: f32) -> f32 {
  %0 = vector.contract #contraction2d_trans_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<3x2xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: func @outerproduct_noacc
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>
// CHECK:      %[[C0:.*]] = constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2xf32>
// CHECK:      %[[T1:.*]] = splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T2:.*]] = mulf %[[T1]], %[[B]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C0]] [0] : vector<3xf32> into vector<2x3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1] : vector<2xf32>
// CHECK:      %[[T5:.*]] = splat %[[T4]] : vector<3xf32>
// CHECK:      %[[T6:.*]] = mulf %[[T5]], %[[B]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xf32> into vector<2x3xf32>
// CHECK:      return %[[T7]] : vector<2x3xf32>

func @outerproduct_noacc(%arg0: vector<2xf32>,
                         %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %0 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %0: vector<2x3xf32>
}

// CHECK-LABEL: func @outerproduct_acc
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2x3xf32>
// CHECK:      %[[C0:.*]] = constant dense<0.000000e+00> : vector<2x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<2xf32>
// CHECK:      %[[T1:.*]] = splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T2:.*]] = vector.extract %[[C]][0] : vector<2x3xf32>
// CHECK:      %[[T3:.*]] = vector.fma %[[T1]], %[[B]], %[[T2]] : vector<3xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[C0]] [0] : vector<3xf32> into vector<2x3xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<2xf32>
// CHECK:      %[[T6:.*]] = splat %[[T5]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.extract %[[C]][1] : vector<2x3xf32>
// CHECK:      %[[T8:.*]] = vector.fma %[[T6]], %[[B]], %[[T7]] : vector<3xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : vector<3xf32> into vector<2x3xf32>
// CHECK:      return %[[T9]] : vector<2x3xf32>

func @outerproduct_acc(%arg0: vector<2xf32>,
                       %arg1: vector<3xf32>,
                       %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %0: vector<2x3xf32>
}

// CHECK-LABEL: func @transpose23
// CHECK-SAME: %[[A:.*]]: vector<2x3xf32>
// CHECK:      %[[Z:.*]] = constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[Z]] [0, 0] : f32 into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.extract %[[A]][1, 0] : vector<2x3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [0, 1] : f32 into vector<3x2xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][0, 1] : vector<2x3xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [1, 0] : f32 into vector<3x2xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[A]][1, 1] : vector<2x3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [1, 1] : f32 into vector<3x2xf32>
// CHECK:      %[[T8:.*]] = vector.extract %[[A]][0, 2] : vector<2x3xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [2, 0] : f32 into vector<3x2xf32>
// CHECK:      %[[T10:.*]] = vector.extract %[[A]][1, 2] : vector<2x3xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [2, 1] : f32 into vector<3x2xf32>
// CHECK:      return %[[T11]] : vector<3x2xf32>

func @transpose23(%arg0: vector<2x3xf32>) -> vector<3x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// Shape up and downcasts for 2-D vectors, for supporting conversion to
// llvm.matrix operations
// CHECK-LABEL: func @shape_casts
func @shape_casts(%a: vector<2x2xf32>) -> (vector<4xf32>, vector<2x2xf32>) {
  // CHECK: %[[cst:.*]] = constant dense<0.000000e+00> : vector<4xf32>
  // CHECK: %[[cst22:.*]] = constant dense<0.000000e+00> : vector<2x2xf32>
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
  // CHECK: %[[add:.*]] = addf %[[in2]], %[[in2]] : vector<4xf32>
  %r0 = addf %0, %0: vector<4xf32>
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

// MATRIX-LABEL: func @matmul
// MATRIX-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// MATRIX-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x3xf32>,
// MATRIX-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      MATRIX:  %[[vcst:.*]] = constant dense<0.000000e+00> : vector<8xf32>
//      MATRIX:  %[[vcst_0:.*]] = constant dense<0.000000e+00> : vector<12xf32>
//      MATRIX:  %[[vcst_1:.*]] = constant dense<0.000000e+00> : vector<2x3xf32>
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
//      MATRIX:  %[[mm6:.*]] = addf %[[C]], %[[mm5]] : vector<2x3xf32>
func @matmul(%arg0: vector<2x4xf32>,
                          %arg1: vector<4x3xf32>,
                          %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @broadcast_vec1d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = splat %[[A]] : vector<2xf32>
// CHECK:      return %[[T0]] : vector<2xf32>

func @broadcast_vec1d_from_scalar(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @broadcast_vec2d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = splat %[[A]] : vector<2x3xf32>
// CHECK:      return %[[T0]] : vector<2x3xf32>

func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = splat %[[A]] : vector<2x3x4xf32>
// CHECK:      return %[[T0]] : vector<2x3x4xf32>

func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_vec1d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      return %[[A]] : vector<2xf32>

func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @broadcast_vec2d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      %[[C0:.*]] = constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[C0]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      return %[[T2]] : vector<3x2xf32>

func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      %[[C0:.*]] = constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[C1:.*]] = constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[C0]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C1]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T2]], %[[T3]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T2]], %[[T4]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T6:.*]] = vector.insert %[[T2]], %[[T5]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:       return %[[T6]] : vector<4x3x2xf32>

func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_vec2d
// CHECK-SAME: %[[A:.*0]]: vector<3x2xf32>
// CHECK:      %[[C0:.*]] = constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[C0]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[A]], %[[T2]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      return %[[T3]] : vector<4x3x2xf32>

func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @broadcast_stretch
// CHECK-SAME: %[[A:.*0]]: vector<1xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<1xf32>
// CHECK:      %[[T1:.*]] = splat %[[T0]] : vector<4xf32>
// CHECK:      return %[[T1]] : vector<4xf32>

func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @broadcast_stretch_at_start
// CHECK-SAME: %[[A:.*0]]: vector<1x4xf32>
// CHECK:      %[[C0:.*]] = constant dense<0.000000e+00> : vector<3x4xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<1x4xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[C0]] [0] : vector<4xf32> into vector<3x4xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[T0]], %[[T1]] [1] : vector<4xf32> into vector<3x4xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T0]], %[[T2]] [2] : vector<4xf32> into vector<3x4xf32>
// CHECK:      return %[[T3]] : vector<3x4xf32>

func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// CHECK-LABEL: func @broadcast_stretch_at_end
// CHECK-SAME: %[[A:.*0]]: vector<4x1xf32>
// CHECK:      %[[C:.*]] = constant dense<0.000000e+00> : vector<4x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<4x1xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[T0]][0] : vector<1xf32>
// CHECK:      %[[T2:.*]] = splat %[[T1]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[C0]] [0] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1] : vector<4x1xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[T4]][0] : vector<1xf32>
// CHECK:      %[[T6:.*]] = splat %[[T5]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T8:.*]] = vector.extract %[[A]][2] : vector<4x1xf32>
// CHECK:      %[[T9:.*]] = vector.extract %[[T8]][0] : vector<1xf32>
// CHECK:      %[[T10:.*]] = splat %[[T9]] : vector<3xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T7]] [2] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][3] : vector<4x1xf32>
// CHECK:      %[[T13:.*]] = vector.extract %[[T12]][0] : vector<1xf32>
// CHECK:      %[[T14:.*]] = splat %[[T13]] : vector<3xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T14]], %[[T11]] [3] : vector<3xf32> into vector<4x3xf32>
// CHECK:      return %[[T15]] : vector<4x3xf32>

func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}

// CHECK-LABEL: func @broadcast_stretch_in_middle
// CHECK-SAME: %[[A:.*0]]: vector<4x1x2xf32>
// CHECK:      %[[C0:.*]] = constant dense<0.000000e+00> : vector<4x3x2xf32>
// CHECK:      %[[C1:.*]] = constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<4x1x2xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[T0]][0] : vector<1x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[T1]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T1]], %[[T2]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T1]], %[[T3]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[C0]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[A]][1] : vector<4x1x2xf32>
// CHECK:      %[[T7:.*]] = vector.extract %[[T6]][0] : vector<1x2xf32>
// CHECK:      %[[T8:.*]] = vector.insert %[[T7]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T7]], %[[T8]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T10:.*]] = vector.insert %[[T7]], %[[T9]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T5]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][2] : vector<4x1x2xf32>
// CHECK:      %[[T13:.*]] = vector.extract %[[T12]][0] : vector<1x2xf32>
// CHECK:      %[[T14:.*]] = vector.insert %[[T13]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T13]], %[[T14]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T16:.*]] = vector.insert %[[T13]], %[[T15]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T17:.*]] = vector.insert %[[T16]], %[[T11]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T18:.*]] = vector.extract %[[A]][3] : vector<4x1x2xf32>
// CHECK:      %[[T19:.*]] = vector.extract %[[T18]][0] : vector<1x2xf32>
// CHECK:      %[[T20:.*]] = vector.insert %[[T19]], %[[C1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T21:.*]] = vector.insert %[[T19]], %[[T20]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T22:.*]] = vector.insert %[[T19]], %[[T21]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T23:.*]] = vector.insert %[[T22]], %[[T17]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      return %[[T23]] : vector<4x3x2xf32>

func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @genbool_1d
// CHECK: %[[TT:.*]] = constant 1 : i1
// CHECK: %[[C1:.*]] = constant dense<false> : vector<8xi1>
// CHECK: %[[T0.*]] = vector.insert %[[TT]], %[[C1]] [0] : i1 into vector<8xi1>
// CHECK: %[[T1.*]] = vector.insert %[[TT]], %[[T0]] [1] : i1 into vector<8xi1>
// CHECK: %[[T2.*]] = vector.insert %[[TT]], %[[T1]] [2] : i1 into vector<8xi1>
// CHECK: %[[T3.*]] = vector.insert %[[TT]], %[[T2]] [3] : i1 into vector<8xi1>
// CHECK: return %[[T3]] : vector<8xi1>

func @genbool_1d() -> vector<8xi1> {
  %0 = vector.constant_mask [4] : vector<8xi1>
  return %0 : vector<8xi1>
}

// CHECK-LABEL: func @genbool_2d
// CHECK: %[[TT:.*]] = constant 1 : i1
// CHECK: %[[C1:.*]] = constant dense<false> : vector<4xi1>
// CHECK: %[[C2:.*]] = constant dense<false> : vector<4x4xi1>
// CHECK: %[[T0:.*]] = vector.insert %[[TT]], %[[C1]] [0] : i1 into vector<4xi1>
// CHECK: %[[T1:.*]] = vector.insert %[[TT]], %[[T0]] [1] : i1 into vector<4xi1>
// CHECK: %[[T2:.*]] = vector.insert %[[T1]], %[[C2]] [0] : vector<4xi1> into vector<4x4xi1>
// CHECK: %[[T3:.*]] = vector.insert %[[T1]], %[[T2]] [1] : vector<4xi1> into vector<4x4xi1>
// CHECK: return %[[T3]] : vector<4x4xi1>

func @genbool_2d() -> vector<4x4xi1> {
  %v = vector.constant_mask [2, 2] : vector<4x4xi1>
  return %v: vector<4x4xi1>
}

// CHECK-LABEL: func @genbool_3d
// CHECK: %[[Tt:.*]] = constant 1 : i1
// CHECK: %[[C1:.*]] = constant dense<false> : vector<4xi1>
// CHECK: %[[C2:.*]] = constant dense<false> : vector<3x4xi1>
// CHECK: %[[C3:.*]] = constant dense<false> : vector<2x3x4xi1>
// CHECK: %[[T0:.*]] = vector.insert %[[TT]], %[[C1]] [0] : i1 into vector<4xi1>
// CHECK: %[[T1:.*]] = vector.insert %[[TT]], %[[T0]] [1] : i1 into vector<4xi1>
// CHECK: %[[T2:.*]] = vector.insert %[[TT]], %[[T1]] [2] : i1 into vector<4xi1>
// CHECK: %[[T3:.*]] = vector.insert %[[T2]], %[[C2]] [0] : vector<4xi1> into vector<3x4xi1>
// CHECK: %[[T4:.*]] = vector.insert %[[T3]], %[[C3]] [0] : vector<3x4xi1> into vector<2x3x4xi1>
// CHECK: return %[[T4]] : vector<2x3x4xi1>

func @genbool_3d() -> vector<2x3x4xi1> {
  %v = vector.constant_mask [1, 1, 3] : vector<2x3x4xi1>
  return %v: vector<2x3x4xi1>
}

// CHECK-LABEL: func @genbool_var_1d
// CHECK-SAME: %[[A:.*0]]: index
// CHECK-DAG:  %[[VF:.*]] = constant dense<false> : vector<3xi1>
// CHECK-DAG:  %[[C0:.*]] = constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = constant 1 : index
// CHECK-DAG:  %[[C2:.*]] = constant 2 : index
// CHECK:      %[[T0:.*]] = cmpi "slt", %[[C0]], %[[A]] : index
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[VF]] [0] : i1 into vector<3xi1>
// CHECK:      %[[T2:.*]] = cmpi "slt", %[[C1]], %[[A]] : index
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [1] : i1 into vector<3xi1>
// CHECK:      %[[T4:.*]] = cmpi "slt", %[[C2]], %[[A]] : index
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [2] : i1 into vector<3xi1>
// CHECK:      return %[[T5]] : vector<3xi1>

func @genbool_var_1d(%arg0: index) -> vector<3xi1> {
  %0 = vector.create_mask %arg0 : vector<3xi1>
  return %0 : vector<3xi1>
}

// CHECK-LABEL: func @genbool_var_2d
// CHECK-SAME: %[[A:.*0]]: index
// CHECK-SAME: %[[B:.*1]]: index
// CHECK-DAG:  %[[Z1:.*]] = constant dense<false> : vector<3xi1>
// CHECK-DAG:  %[[Z2:.*]] = constant dense<false> : vector<2x3xi1>
// CHECK-DAG:  %[[C0:.*]] = constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = constant 1 : index
// CHECK-DAG:  %[[C2:.*]] = constant 2 : index
// CHECK:      %[[T0:.*]] = cmpi "slt", %[[C0]], %[[B]] : index
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[Z1]] [0] : i1 into vector<3xi1>
// CHECK:      %[[T2:.*]] = cmpi "slt", %[[C1]], %[[B]] : index
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [1] : i1 into vector<3xi1>
// CHECK:      %[[T4:.*]] = cmpi "slt", %[[C2]], %[[B]] : index
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [2] : i1 into vector<3xi1>
// CHECK:      %[[T6:.*]] = cmpi "slt", %[[C0]], %[[A]] : index
// CHECK:      %[[T7:.*]] = select %[[T6]], %[[T5]], %[[Z1]] : vector<3xi1>
// CHECK:      %[[T8:.*]] = vector.insert %7, %[[Z2]] [0] : vector<3xi1> into vector<2x3xi1>
// CHECK:      %[[T9:.*]] = cmpi "slt", %[[C1]], %[[A]] : index
// CHECK:      %[[T10:.*]] = select %[[T9]], %[[T5]], %[[Z1]] : vector<3xi1>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T8]] [1] : vector<3xi1> into vector<2x3xi1>
// CHECK:      return %[[T11]] : vector<2x3xi1>

func @genbool_var_2d(%arg0: index, %arg1: index) -> vector<2x3xi1> {
  %0 = vector.create_mask %arg0, %arg1 : vector<2x3xi1>
  return %0 : vector<2x3xi1>
}
