// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" %s -o -| FileCheck %s

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @broadcast_resize_nearest_fp
func.func @broadcast_resize_nearest_fp(%arg0 : tensor<3x1x1x7xf32>) -> tensor<3x15x13x7xf32> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-SAME{literal}: [[0], [1, 2, 3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x15x13x7xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xf32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<3x15x13x7xf32>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK:   linalg.yield %[[IN]]
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [2, 2, 1, 1], offset = [0, 0], border = [0, 0]} : (tensor<3x1x1x7xf32>) -> tensor<3x15x13x7xf32>

  // CHECK: return %[[GENERIC]]
  return %resize : tensor<3x15x13x7xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @broadcast_resize_bilinear_fp
func.func @broadcast_resize_bilinear_fp(%arg0 : tensor<3x1x1x7xf32>) -> tensor<3x15x13x7xf32> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-SAME{literal}: [[0], [1, 2, 3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x15x13x7xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xf32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<3x15x13x7xf32>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK:   linalg.yield %[[IN]]
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [2, 2, 1, 1], offset = [0, 0], border = [0, 0]} : (tensor<3x1x1x7xf32>) -> tensor<3x15x13x7xf32>

  // CHECK: return %[[GENERIC]]
  return %resize : tensor<3x15x13x7xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @broadcast_resize_nearest_i8
func.func @broadcast_resize_nearest_i8(%arg0 : tensor<3x1x1x7xi8>) -> tensor<3x15x13x7xi8> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-SAME{literal}: [[0], [1, 2, 3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x15x13x7xi8>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xi8>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<3x15x13x7xi8>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: i8, %[[OUT:.+]]: i8):
  // CHECK:   linalg.yield %[[IN]]
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [2, 2, 1, 1], offset = [0, 0], border = [0, 0]} : (tensor<3x1x1x7xi8>) -> tensor<3x15x13x7xi8>

  // CHECK: return %[[GENERIC]]
  return %resize : tensor<3x15x13x7xi8>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @broadcast_resize_nearest_i32
func.func @broadcast_resize_nearest_i32(%arg0 : tensor<3x1x1x7xi8>) -> tensor<3x15x13x7xi32> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-SAME{literal}: [[0], [1, 2, 3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x15x13x7xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xi8>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<3x15x13x7xi32>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: i8, %[[OUT:.+]]: i32):
  // CHECK:   %[[EXT:.+]] = arith.extsi %[[IN]] : i8 to i32
  // CHECK:   linalg.yield %[[EXT]]
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [2, 2, 1, 1], offset = [0, 0], border = [0, 0]} : (tensor<3x1x1x7xi8>) -> tensor<3x15x13x7xi32>

  // CHECK: return %[[GENERIC]]
  return %resize : tensor<3x15x13x7xi32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @broadcast_resize_bilinear_i32
func.func @broadcast_resize_bilinear_i32(%arg0 : tensor<3x1x1x7xi8>) -> tensor<3x15x13x7xi32> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-SAME{literal}: [[0], [1, 2, 3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x15x13x7xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xi8>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<3x15x13x7xi32>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: i8, %[[OUT:.+]]: i32):
  // CHECK: %[[EXT:.+]] = arith.extsi %[[IN]] : i8 to i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK: %[[MUL1:.+]] = arith.muli %[[EXT]], %[[C2]] : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK: %[[MUL2:.+]] = arith.muli %[[MUL1]], %[[C1]] : i32
  // CHECK: linalg.yield %[[MUL2]]
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [2, 2, 1, 1], offset = [0, 0], border = [0, 0]} : (tensor<3x1x1x7xi8>) -> tensor<3x15x13x7xi32>

  // CHECK: return %[[GENERIC]]
  return %resize : tensor<3x15x13x7xi32>
}

// -----

// CHECK-LABEL:  @resize_nearest_int
func.func @resize_nearest_int(%arg0: tensor<1x15x13x1xi8>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x23x179x1xi8>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK-DAG: %[[XY_MIN:.+]] = arith.constant 0
  // CHECK-DAG: %[[Y_MAX:.+]] = arith.constant 14
  // CHECK-DAG: %[[X_MAX:.+]] = arith.constant 12

  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX_1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX_2]]
  // CHECK-DAG: %[[SCALE_Y_N:.*]] = arith.constant 11
  // CHECK-DAG: %[[SCALE_Y_D:.*]] = arith.constant 7
  // CHECK-DAG: %[[SCALE_X_N:.*]] = arith.constant 89
  // CHECK-DAG: %[[SCALE_X_D:.*]] = arith.constant 6
  // CHECK-DAG: %[[OFFSET_Y:.*]] = arith.constant 0
  // CHECK-DAG: %[[OFFSET_X:.*]] = arith.constant 0
  // CHECK-DAG: %[[BORDER_Y:.*]] = arith.constant 0
  // CHECK-DAG: %[[BORDER_X:.*]] = arith.constant 0

  // find the remainder and integer component of the target index.

  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[Y]], %[[SCALE_Y_D]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[X]], %[[SCALE_X_D]]
  // CHECK: %[[Y:.*]] = arith.addi %[[TEMP_Y]], %[[OFFSET_Y]]
  // CHECK: %[[X:.*]] = arith.addi %[[TEMP_X]], %[[OFFSET_X]]
  // CHECK: %[[I_Y:.*]] = arith.divsi %[[Y]], %[[SCALE_Y_N]]
  // CHECK: %[[I_X:.*]] = arith.divsi %[[X]], %[[SCALE_X_N]]
  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[I_Y]], %[[SCALE_Y_N]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[I_X]], %[[SCALE_X_N]]
  // CHECK: %[[D_Y:.*]] = arith.subi %[[Y]], %[[TEMP_Y]]
  // CHECK: %[[D_X:.*]] = arith.subi %[[X]], %[[TEMP_X]]

  // Round to the nearest neighor.

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[D_Y_DOUBLE:.*]] = arith.shli %[[D_Y]], %[[ONE]]
  // CHECK: %[[D_X_DOUBLE:.*]] = arith.shli %[[D_X]], %[[ONE]]
  // CHECK: %[[PRED_Y:.*]] = arith.cmpi sge, %[[D_Y_DOUBLE]], %[[SCALE_Y_N]]
  // CHECK: %[[PRED_X:.*]] = arith.cmpi sge, %[[D_X_DOUBLE]], %[[SCALE_X_N]]
  // CHECK: %[[VAL_37:.*]] = arith.select %[[PRED_Y]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_38:.*]] = arith.select %[[PRED_X]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_39:.*]] = arith.addi %[[I_Y]], %[[VAL_37]]
  // CHECK: %[[VAL_40:.*]] = arith.addi %[[I_X]], %[[VAL_38]]

  // This section applies bound checking to be within the input image.

  // CHECK: %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_39]], %[[XY_MIN]]
  // CHECK: %[[VAL_42:.*]] = arith.select %[[VAL_41]], %[[XY_MIN]], %[[VAL_39]]
  // CHECK: %[[VAL_43:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[VAL_39]]
  // CHECK: %[[VAL_44:.*]] = arith.select %[[VAL_43]], %[[Y_MAX]], %[[VAL_42]]
  // CHECK: %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_40]], %[[XY_MIN]]
  // CHECK: %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[XY_MIN]], %[[VAL_40]]
  // CHECK: %[[VAL_47:.*]] = arith.cmpi slt, %[[X_MAX]], %[[VAL_40]]
  // CHECK: %[[VAL_48:.*]] = arith.select %[[VAL_47]], %[[X_MAX]], %[[VAL_46]]

  // Extract the nearest value using the computed indices.

  // CHECK: %[[IDY:.+]] = arith.index_cast %[[VAL_44]]
  // CHECK: %[[IDX:.+]] = arith.index_cast %[[VAL_48]]
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[IDY]], %[[IDX]], %[[IDX_3]]]
  // CHECK: linalg.yield %[[EXTRACT]]

  // Round to the nearest index.
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [11, 7, 89, 6], offset = [0, 0], border = [0, 0]} : (tensor<1x15x13x1xi8>) -> tensor<1x23x179x1xi8>
           return
}

// -----

// CHECK-LABEL:  @resize_bilinear_int
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @resize_bilinear_int(%arg0: tensor<1x19x19x1xi8>) {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x289x289x1xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK-DAG: %[[XY_MIN:.+]] = arith.constant 0
  // CHECK-DAG: %[[Y_MAX:.+]] = arith.constant 18
  // CHECK-DAG: %[[X_MAX:.+]] = arith.constant 18
  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX_1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX_2]]
  // CHECK-DAG: %[[SCALE_Y_N:.*]] = arith.constant 16
  // CHECK-DAG: %[[SCALE_Y_D:.*]] = arith.constant 1
  // CHECK-DAG: %[[SCALE_X_N:.*]] = arith.constant 16
  // CHECK-DAG: %[[SCALE_X_D:.*]] = arith.constant 1
  // CHECK-DAG: %[[OFFSET_Y:.*]] = arith.constant 0
  // CHECK-DAG: %[[OFFSET_X:.*]] = arith.constant 0
  // CHECK-DAG: %[[BORDER_Y:.*]] = arith.constant 0
  // CHECK-DAG: %[[BORDER_X:.*]] = arith.constant 0

  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[Y]], %[[SCALE_Y_D]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[X]], %[[SCALE_X_D]]
  // CHECK: %[[Y:.*]] = arith.addi %[[TEMP_Y]], %[[OFFSET_Y]]
  // CHECK: %[[X:.*]] = arith.addi %[[TEMP_X]], %[[OFFSET_X]]
  // CHECK: %[[I_Y:.*]] = arith.divsi %[[Y]], %[[SCALE_Y_N]]
  // CHECK: %[[I_X:.*]] = arith.divsi %[[X]], %[[SCALE_X_N]]
  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[I_Y]], %[[SCALE_Y_N]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[I_X]], %[[SCALE_X_N]]
  // CHECK: %[[D_Y:.*]] = arith.subi %[[Y]], %[[TEMP_Y]]
  // CHECK: %[[D_X:.*]] = arith.subi %[[X]], %[[TEMP_X]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[Y1:.*]] = arith.addi %[[I_Y]], %[[ONE]]
  // CHECK: %[[X1:.*]] = arith.addi %[[I_X]], %[[ONE]]

  // Bound check each dimension.

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_Y]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_Y]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[I_Y]]
  // CHECK: %[[YLO:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[Y1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[Y1]]
  // CHECK: %[[YHI:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_X]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_X]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[I_X]]
  // CHECK: %[[XLO:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[X1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[X1]]
  // CHECK: %[[XHI:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // Extract each corner of the bilinear interpolation.

  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]
  // CHECK: %[[XLOI:.+]] = arith.index_cast %[[XLO]]
  // CHECK: %[[XHII:.+]] = arith.index_cast %[[XHI]]

  // CHECK: %[[LOLO:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YLOI]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[LOHI:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YLOI]], %[[XHII]], %[[IDX_3]]]
  // CHECK: %[[HILO:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YHII]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[HIHI:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YHII]], %[[XHII]], %[[IDX_3]]]

  // CHECK: %[[XLOLO:.+]] = arith.extsi %[[LOLO]]
  // CHECK: %[[XLOHI:.+]] = arith.extsi %[[LOHI]]
  // CHECK: %[[XHILO:.+]] = arith.extsi %[[HILO]]
  // CHECK: %[[XHIHI:.+]] = arith.extsi %[[HIHI]]

  // Compute the bilinear interpolation.

  // CHECK: %[[NDX:.+]] = arith.subi %[[SCALE_X_N]], %[[D_X]]
  // CHECK: %[[WLOLO:.+]] = arith.muli %[[XLOLO]], %[[NDX]]
  // CHECK: %[[WLOHI:.+]] = arith.muli %[[XLOHI]], %[[D_X]]
  // CHECK: %[[LO:.+]] = arith.addi %[[WLOLO]], %[[WLOHI]]
  // CHECK: %[[WHILO:.+]] = arith.muli %[[XHILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.muli %[[XHIHI]], %[[D_X]]
  // CHECK: %[[HI:.+]] = arith.addi %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subi %[[SCALE_Y_N]], %[[D_Y]]
  // CHECK: %[[WLO:.+]] = arith.muli %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.muli %[[HI]], %[[D_Y]]
  // CHECK: %[[RESULT:.+]] = arith.addi %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]

  // Round to the nearest index.
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [16, 1, 16, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x19x19x1xi8>) -> tensor<1x289x289x1xi32>
           return
}

// -----

// CHECK-LABEL: @resize_nearest_fp
func.func @resize_nearest_fp(%input: tensor<1x50x48x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x1600x1536x1xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX0:.+]] = linalg.index 0
  // CHECK: %[[IDX1:.+]] = linalg.index 1
  // CHECK: %[[IDX2:.+]] = linalg.index 2
  // CHECK: %[[IDX3:.+]] = linalg.index 3
  // CHECK-DAG: %[[XYMIN:.*]] = arith.constant 0
  // CHECK-DAG: %[[YMAX:.*]] = arith.constant 49
  // CHECK-DAG: %[[XMAX:.*]] = arith.constant 47
  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX2]]
  // CHECK-DAG: %[[ISCALE_Y_N:.*]] = arith.constant 64
  // CHECK-DAG: %[[ISCALE_Y_D:.*]] = arith.constant 2
  // CHECK-DAG: %[[ISCALE_X_N:.*]] = arith.constant 64
  // CHECK-DAG: %[[ISCALE_X_D:.*]] = arith.constant 2
  // CHECK-DAG: %[[IOFFSET_Y:.*]] = arith.constant -31
  // CHECK-DAG: %[[IOFFSET_X:.*]] = arith.constant -31
  // CHECK-DAG: %[[IBORDER_Y:.*]] = arith.constant 31
  // CHECK-DAG: %[[IBORDER_X:.*]] = arith.constant 31

  // CHECK: %[[Y0:.+]] = arith.uitofp %[[Y]]
  // CHECK: %[[X0:.+]] = arith.uitofp %[[X]]
  // CHECK: %[[SCALE_Y_N:.*]] = arith.uitofp %[[ISCALE_Y_N]]
  // CHECK: %[[SCALE_Y_D:.*]] = arith.uitofp %[[ISCALE_Y_D]]
  // CHECK: %[[SCALE_X_N:.*]] = arith.uitofp %[[ISCALE_X_N]]
  // CHECK: %[[SCALE_X_D:.*]] = arith.uitofp %[[ISCALE_X_D]]
  // CHECK: %[[OFFSET_Y:.*]] = arith.uitofp %[[IOFFSET_Y]]
  // CHECK: %[[OFFSET_X:.*]] = arith.uitofp %[[IOFFSET_X]]

  // CHECK: %[[VAL_29:.*]] = arith.mulf %[[Y0]], %[[SCALE_Y_D]]
  // CHECK: %[[VAL_30:.*]] = arith.mulf %[[X0]], %[[SCALE_X_D]]
  // CHECK: %[[VAL_31:.*]] = arith.addf %[[VAL_29]], %[[OFFSET_Y]]
  // CHECK: %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[OFFSET_X]]
  // CHECK: %[[VAL_33:.*]] = arith.divf %[[VAL_31]], %[[SCALE_Y_N]]
  // CHECK: %[[VAL_34:.*]] = arith.divf %[[VAL_32]], %[[SCALE_X_N]]

  // Find the remainder and integer component of the target index.

  // CHECK: %[[VAL_35:.*]] = math.floor %[[VAL_33]]
  // CHECK: %[[VAL_36:.*]] = math.floor %[[VAL_34]]
  // CHECK: %[[D_Y:.*]] = arith.subf %[[VAL_33]], %[[VAL_35]]
  // CHECK: %[[D_X:.*]] = arith.subf %[[VAL_34]], %[[VAL_36]]
  // CHECK: %[[VAL_39:.*]] = arith.fptosi %[[VAL_35]]
  // CHECK: %[[VAL_40:.*]] = arith.fptosi %[[VAL_36]]

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01
  // CHECK: %[[PRED_Y:.*]] = arith.cmpf oge, %[[D_Y]], %[[HALF]]
  // CHECK: %[[PRED_X:.*]] = arith.cmpf oge, %[[D_X]], %[[HALF]]
  // CHECK: %[[ROUND_Y:.*]] = arith.select %[[PRED_Y]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[ROUND_X:.*]] = arith.select %[[PRED_X]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_48:.*]] = arith.addi %[[VAL_39]], %[[ROUND_Y]]
  // CHECK: %[[VAL_49:.*]] = arith.addi %[[VAL_40]], %[[ROUND_X]]

  // CHECK: %[[VAL_50:.*]] = arith.cmpi slt, %[[VAL_48]], %[[XYMIN]]
  // CHECK: %[[VAL_51:.*]] = arith.select %[[VAL_50]], %[[XYMIN]], %[[VAL_48]]
  // CHECK: %[[VAL_52:.*]] = arith.cmpi slt, %[[YMAX]], %[[VAL_48]]
  // CHECK: %[[VAL_53:.*]] = arith.select %[[VAL_52]], %[[YMAX]], %[[VAL_51]]
  // CHECK: %[[VAL_54:.*]] = arith.cmpi slt, %[[VAL_49]], %[[XYMIN]]
  // CHECK: %[[VAL_55:.*]] = arith.select %[[VAL_54]], %[[XYMIN]], %[[VAL_49]]
  // CHECK: %[[VAL_56:.*]] = arith.cmpi slt, %[[XMAX]], %[[VAL_49]]
  // CHECK: %[[VAL_57:.*]] = arith.select %[[VAL_56]], %[[XMAX]], %[[VAL_55]]

  // CHECK: %[[IDY:.*]] = arith.index_cast %[[VAL_53]]
  // CHECK: %[[IDX:.*]] = arith.index_cast %[[VAL_57]]
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX0]], %[[IDY]], %[[IDX]], %[[IDX3]]]
  // CHECK: linalg.yield %[[EXTRACT]]

  %output = "tosa.resize"(%input) {mode = "NEAREST_NEIGHBOR", scale = [64, 2, 64, 2], offset = [-31, -31], border = [31, 31]} : (tensor<1x50x48x1xf32>) -> tensor<1x1600x1536x1xf32>

  return
}

// -----

// CHECK-LABEL: @resize_bilinear_fp
func.func @resize_bilinear_fp(%input: tensor<1x23x23x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x89x89x1xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK-DAG: %[[XY_MIN:.*]] = arith.constant 0
  // CHECK-DAG: %[[Y_MAX:.*]] = arith.constant 22
  // CHECK-DAG: %[[X_MAX:.*]] = arith.constant 22
  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX_1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX_2]]
  // CHECK-DAG: %[[ISCALE_Y_N:.*]] = arith.constant 4
  // CHECK-DAG: %[[ISCALE_Y_D:.*]] = arith.constant 1
  // CHECK-DAG: %[[ISCALE_X_N:.*]] = arith.constant 4
  // CHECK-DAG: %[[ISCALE_X_D:.*]] = arith.constant 1
  // CHECK-DAG: %[[IOFFSET_Y:.*]] = arith.constant 0
  // CHECK-DAG: %[[IOFFSET_X:.*]] = arith.constant 0
  // CHECK-DAG: %[[IBORDER_Y:.*]] = arith.constant 0
  // CHECK-DAG: %[[IBORDER_X:.*]] = arith.constant 0

  // CHECK: %[[Y0:.+]] = arith.uitofp %[[Y]]
  // CHECK: %[[X0:.+]] = arith.uitofp %[[X]]
  // CHECK: %[[SCALE_Y_N:.*]] = arith.uitofp %[[ISCALE_Y_N]]
  // CHECK: %[[SCALE_Y_D:.*]] = arith.uitofp %[[ISCALE_Y_D]]
  // CHECK: %[[SCALE_X_N:.*]] = arith.uitofp %[[ISCALE_X_N]]
  // CHECK: %[[SCALE_X_D:.*]] = arith.uitofp %[[ISCALE_X_D]]
  // CHECK: %[[OFFSET_Y:.*]] = arith.uitofp %[[IOFFSET_Y]]
  // CHECK: %[[OFFSET_X:.*]] = arith.uitofp %[[IOFFSET_X]]

  // CHECK: %[[VAL_29:.*]] = arith.mulf %[[Y0]], %[[SCALE_Y_D]]
  // CHECK: %[[VAL_30:.*]] = arith.mulf %[[X0]], %[[SCALE_X_D]]
  // CHECK: %[[VAL_31:.*]] = arith.addf %[[VAL_29]], %[[OFFSET_Y]]
  // CHECK: %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[OFFSET_X]]
  // CHECK: %[[VAL_33:.*]] = arith.divf %[[VAL_31]], %[[SCALE_Y_N]]
  // CHECK: %[[VAL_34:.*]] = arith.divf %[[VAL_32]], %[[SCALE_X_N]]

  // CHECK: %[[VAL_35:.*]] = math.floor %[[VAL_33]]
  // CHECK: %[[VAL_36:.*]] = math.floor %[[VAL_34]]
  // CHECK: %[[D_Y:.*]] = arith.subf %[[VAL_33]], %[[VAL_35]]
  // CHECK: %[[D_X:.*]] = arith.subf %[[VAL_34]], %[[VAL_36]]
  // CHECK: %[[I_Y:.*]] = arith.fptosi %[[VAL_35]]
  // CHECK: %[[I_X:.*]] = arith.fptosi %[[VAL_36]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[Y1:.*]] = arith.addi %[[I_Y]], %[[ONE]]
  // CHECK: %[[X1:.*]] = arith.addi %[[I_X]], %[[ONE]]

  // Bound check each dimension.

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_Y]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_Y]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[I_Y]]
  // CHECK: %[[YLO:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[Y1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[Y1]]
  // CHECK: %[[YHI:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_X]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_X]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[I_X]]
  // CHECK: %[[XLO:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[X1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[X1]]
  // CHECK: %[[XHI:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]
  // CHECK: %[[XLOI:.+]] = arith.index_cast %[[XLO]]
  // CHECK: %[[XHII:.+]] = arith.index_cast %[[XHI]]

  // CHECK: %[[LOLO:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YLOI]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[LOHI:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YLOI]], %[[XHII]], %[[IDX_3]]]
  // CHECK: %[[HILO:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YHII]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[HIHI:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YHII]], %[[XHII]], %[[IDX_3]]]

  // CHECK-DAG: %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[NDX:.+]] = arith.subf %[[ONE]], %[[D_X]]
  // CHECK: %[[WLOLO:.+]] = arith.mulf %[[LOLO]], %[[NDX]]
  // CHECK: %[[WLOHI:.+]] = arith.mulf %[[LOHI]], %[[D_X]]
  // CHECK: %[[LO:.+]] = arith.addf %[[WLOLO]], %[[WLOHI]]
  // CHECK: %[[WHILO:.+]] = arith.mulf %[[HILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.mulf %[[HIHI]], %[[D_X]]
  // CHECK: %[[HI:.+]] = arith.addf %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subf %[[ONE]], %[[D_Y]]
  // CHECK: %[[WLO:.+]] = arith.mulf %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.mulf %[[HI]], %[[D_Y]]
  // CHECK: %[[RESULT:.+]] = arith.addf %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]

  // Round by bilinear interpolation
  %output = "tosa.resize"(%input) {mode = "BILINEAR", scale = [4, 1, 4, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x23x23x1xf32>) -> tensor<1x89x89x1xf32>

  return
}

// -----

// CHECK-LABEL: @resize_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @resize_dyn(%input: tensor<?x2x2x1xi8>) -> () {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]]) : tensor<?x4x4x1xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  %output = "tosa.resize"(%input) { scale = [4, 2, 4, 2], offset = [-1, -1], border = [1, 1], mode = "BILINEAR" } : (tensor<?x2x2x1xi8>)  -> (tensor<?x4x4x1xi32>)
  return
}

// -----

// CHECK-LABEL: @resize_bilinear_int48
func.func @resize_bilinear_int48(%arg0: tensor<1x19x19x1xi16>) {
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [16, 1, 16, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x19x19x1xi16>) -> tensor<1x289x289x1xi48>
           return
}
