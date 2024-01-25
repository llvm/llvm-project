// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" %s -o -| FileCheck %s

// CHECK-LABEL: @unary_resize_nearest_fp32
func.func @unary_resize_nearest_fp32(%arg0 : tensor<3x1x1x7xf32>) -> tensor<3x1x1x7xf32> {
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = array<i64: 2, 2, 1, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xf32>) -> tensor<3x1x1x7xf32>
  // CHECK: return %arg0
  return %resize : tensor<3x1x1x7xf32>
}

// -----

// CHECK-LABEL: @unary_resize_nearest_fp16
func.func @unary_resize_nearest_fp16(%arg0 : tensor<3x1x1x7xf16>) -> tensor<3x1x1x7xf16> {
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = array<i64: 2, 2, 1, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xf16>) -> tensor<3x1x1x7xf16>
  // CHECK: return %arg0
  return %resize : tensor<3x1x1x7xf16>
}

// -----

// CHECK-LABEL: @unary_resize_bilinear_fp32
func.func @unary_resize_bilinear_fp32(%arg0 : tensor<3x1x1x7xf32>) -> tensor<3x1x1x7xf32> {
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 2, 2, 1, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xf32>) -> tensor<3x1x1x7xf32>
  // CHECK: return %arg0
  return %resize : tensor<3x1x1x7xf32>
}

// -----

// CHECK-LABEL: @unary_resize_bilinear_fp16
func.func @unary_resize_bilinear_fp16(%arg0 : tensor<3x1x1x7xf16>) -> tensor<3x1x1x7xf16> {
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 2, 2, 1, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xf16>) -> tensor<3x1x1x7xf16>
  // CHECK: return %arg0
  return %resize : tensor<3x1x1x7xf16>
}

// -----

// CHECK-LABEL: @unary_resize_nearest_i8
func.func @unary_resize_nearest_i8(%arg0 : tensor<3x1x1x7xi8>) -> tensor<3x1x1x7xi8> {
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = array<i64: 2, 1, 3, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xi8>) -> tensor<3x1x1x7xi8>
  // CHECK: return %arg0
  return %resize : tensor<3x1x1x7xi8>
}

// -----

// CHECK-LABEL: @broadcast_resize_nearest_f32
func.func @broadcast_resize_nearest_f32(%arg0 : tensor<3x1x1x7xf32>) -> tensor<3x1x5x7xf32> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-NEXT{literal}: [[0], [1, 2, 3]] : tensor<3x1x1x7xf32> into tensor<3x7xf32>
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x1x5x7xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic 
  // CHECK-SAME: indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xf32>) outs(%[[EMPTY]] : tensor<3x1x5x7xf32>)
  // CHECK: ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
  // CHECK:   linalg.yield %[[IN]] : f32
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = array<i64: 2, 1, 3, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xf32>) -> tensor<3x1x5x7xf32>

 // CHECK: return %[[GENERIC]]
  return %resize : tensor<3x1x5x7xf32>
}

// -----

// CHECK-LABEL: @broadcast_resize_bilinear_i8
func.func @broadcast_resize_bilinear_i8(%arg0 : tensor<3x1x1x7xi8>) -> tensor<3x4x5x7xi32> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-SAME{literal}: [[0], [1, 2, 3]] : tensor<3x1x1x7xi8> into tensor<3x7xi8>
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x7xi32>
  // CHECK: %[[RESIZE:.+]] = linalg.generic
  // CHECK-SAME: {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xi8>) outs(%[[EMPTY]] : tensor<3x7xi32>)
  // CHECK: ^bb0(%[[IN:.+]]: i8, %[[OUT:.+]]: i32):
  // CHECK:   %[[EXT:.+]] = arith.extsi %[[IN]] : i8 to i32
  // CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : i32
  // CHECK:   %[[MUL:.+]] = arith.muli %[[EXT]], %[[C2]] : i32
  // CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : i32
  // CHECK:   %[[OUT:.+]] = arith.muli %[[MUL]], %[[C3]] : i32
  // CHECK:   linalg.yield %[[OUT]] : i32
  // CHECK: } -> tensor<3x7xi32>
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %1
  // CHECK-SAME{literal}: [[0], [1, 2, 3]] : tensor<3x7xi32> into tensor<3x1x1x7xi32>
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %expanded
  // CHECK-SAME{literal}:[[0], [1, 2, 3]] : tensor<3x1x1x7xi32> into tensor<3x7xi32>
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x4x5x7xi32>
  // CHECK: %[[BROADCAST:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xi32>) outs(%[[EMPTY]] : tensor<3x4x5x7xi32>) {
  // CHECK: ^bb0(%[[IN:.+]]: i32, %[[OUT:.+]]: i32):
  // CHECK:   linalg.yield %[[IN]] : i32
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 2, 1, 3, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xi8>) -> tensor<3x4x5x7xi32>

  // CHECK: return %[[BROADCAST]]
  return %resize : tensor<3x4x5x7xi32>
}

// -----

// CHECK-LABEL: @unary_resize_bilinear_i32
func.func @unary_resize_bilinear_i32(%arg0 : tensor<3x1x1x7xi8>) -> tensor<3x1x1x7xi32> {
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0
  // CHECK-SAME{literal}: [[0], [1, 2, 3]] : tensor<3x1x1x7xi8> into tensor<3x7xi8>
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x7xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic 
  // CHECK-SAME: indexing_maps = [#map, #map]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]}
  // CHECK-SAME: ins(%[[COLLAPSE]] : tensor<3x7xi8>) outs(%[[EMPTY]] : tensor<3x7xi32>) {
  // CHECK: ^bb0(%[[IN:.+]]: i8, %[[OUT:.+]]: i32):
  // CHECK:   %[[EXT:.+]] = arith.extsi %[[IN]] : i8 to i32
  // CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : i32
  // CHECK:   %[[MUL0:.+]] = arith.muli %[[EXT]], %[[C2]] : i32
  // CHECK-DAG:   %[[C1:.+]] = arith.constant 2 : i32
  // CHECK:   %4 = arith.muli %3, %[[C1]] : i32
  // CHECK:   linalg.yield %4 : i32
  // CHECK: } -> tensor<3x7xi32>
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[GENERIC:.+]]
  // CHECK-SAME{literal} [[0], [1, 2, 3]] : tensor<3x7xi32> into tensor<3x1x1x7xi32>
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 2, 1, 2, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x1x7xi8>) -> tensor<3x1x1x7xi32>

  // CHECK: return %[[EXPAND]]
  return %resize : tensor<3x1x1x7xi32>
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
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant 0
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
  // CHECK: %[[Y:.*]] = arith.addi %[[TEMP_Y]], %[[OFFSET_Y]]
  // CHECK: %[[I_Y:.*]] = arith.divsi %[[Y]], %[[SCALE_Y_N]]
  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[I_Y]], %[[SCALE_Y_N]]
  // CHECK: %[[D_Y:.*]] = arith.subi %[[Y]], %[[TEMP_Y]]

  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[X]], %[[SCALE_X_D]]
  // CHECK: %[[X:.*]] = arith.addi %[[TEMP_X]], %[[OFFSET_X]]
  // CHECK: %[[I_X:.*]] = arith.divsi %[[X]], %[[SCALE_X_N]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[I_X]], %[[SCALE_X_N]]
  // CHECK: %[[D_X:.*]] = arith.subi %[[X]], %[[TEMP_X]]

  // Compute the offset and bound for the Y position.
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[D_Y_DOUBLE:.*]] = arith.shli %[[D_Y]], %[[ONE]]
  // CHECK: %[[PRED_Y:.*]] = arith.cmpi sge, %[[D_Y_DOUBLE]], %[[SCALE_Y_N]]
  // CHECK: %[[VAL_37:.*]] = arith.select %[[PRED_Y]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_39:.*]] = arith.addi %[[I_Y]], %[[VAL_37]]
  // CHECK: %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_39]], %[[ZERO]]
  // CHECK: %[[VAL_42:.*]] = arith.select %[[VAL_41]], %[[ZERO]], %[[VAL_39]]
  // CHECK: %[[VAL_43:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[VAL_39]]
  // CHECK: %[[VAL_44:.*]] = arith.select %[[VAL_43]], %[[Y_MAX]], %[[VAL_42]]
  // CHECK: %[[IDY:.+]] = arith.index_cast %[[VAL_44]]

  // Compute the offset and bound for the X position.
  // CHECK: %[[D_X_DOUBLE:.*]] = arith.shli %[[D_X]], %[[ONE]]
  // CHECK: %[[PRED_X:.*]] = arith.cmpi sge, %[[D_X_DOUBLE]], %[[SCALE_X_N]]
  // CHECK: %[[VAL_38:.*]] = arith.select %[[PRED_X]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_40:.*]] = arith.addi %[[I_X]], %[[VAL_38]]
  // CHECK: %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_40]], %[[ZERO]]
  // CHECK: %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[ZERO]], %[[VAL_40]]
  // CHECK: %[[VAL_47:.*]] = arith.cmpi slt, %[[X_MAX]], %[[VAL_40]]
  // CHECK: %[[VAL_48:.*]] = arith.select %[[VAL_47]], %[[X_MAX]], %[[VAL_46]]
  // CHECK: %[[IDX:.+]] = arith.index_cast %[[VAL_48]]

  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[IDY]], %[[IDX]], %[[IDX_3]]]
  // CHECK: linalg.yield %[[EXTRACT]]

  // Round to the nearest index.
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = array<i64: 11, 7, 89, 6>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x15x13x1xi8>) -> tensor<1x23x179x1xi8>
  return
}

// -----

// CHECK-LABEL:  @resize_bilinear_int
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @resize_bilinear_int(%arg0: tensor<1x19x20x1xi8>) {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x304x320x1xi48>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant 0
  // CHECK-DAG: %[[Y_MAX:.+]] = arith.constant 18
  // CHECK-DAG: %[[X_MAX:.+]] = arith.constant 19
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
  // CHECK: %[[Y:.*]] = arith.addi %[[TEMP_Y]], %[[OFFSET_Y]]
  // CHECK: %[[I_Y:.*]] = arith.divsi %[[Y]], %[[SCALE_Y_N]]
  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[I_Y]], %[[SCALE_Y_N]]
  // CHECK: %[[D_Y:.*]] = arith.subi %[[Y]], %[[TEMP_Y]]

  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[X]], %[[SCALE_X_D]]
  // CHECK: %[[X:.*]] = arith.addi %[[TEMP_X]], %[[OFFSET_X]]
  // CHECK: %[[I_X:.*]] = arith.divsi %[[X]], %[[SCALE_X_N]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[I_X]], %[[SCALE_X_N]]
  // CHECK: %[[D_X:.*]] = arith.subi %[[X]], %[[TEMP_X]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[Y1:.*]] = arith.addi %[[I_Y]], %[[ONE]]

  // Bound check each dimension.

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_Y]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[I_Y]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[I_Y]]
  // CHECK: %[[YLO:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y1]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[Y1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[Y1]]
  // CHECK: %[[YHI:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]

  // CHECK: %[[X1:.*]] = arith.addi %[[I_X]], %[[ONE]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_X]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[I_X]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[I_X]]
  // CHECK: %[[XLO:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X1]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[X1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[X1]]
  // CHECK: %[[XHI:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[XLOI:.+]] = arith.index_cast %[[XLO]]
  // CHECK: %[[XHII:.+]] = arith.index_cast %[[XHI]]

  // Extract each corner of the bilinear interpolation.

  // CHECK: %[[LOLO:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YLOI]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[LOHI:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YLOI]], %[[XHII]], %[[IDX_3]]]
  // CHECK: %[[HILO:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YHII]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[HIHI:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YHII]], %[[XHII]], %[[IDX_3]]]

  // CHECK: %[[XLOLO:.+]] = arith.extsi %[[LOLO]]
  // CHECK: %[[XLOHI:.+]] = arith.extsi %[[LOHI]]
  // CHECK: %[[XHILO:.+]] = arith.extsi %[[HILO]]
  // CHECK: %[[XHIHI:.+]] = arith.extsi %[[HIHI]]

  // CHECK-NEXT: %[[D_X_EXT:.+]] = arith.extsi %[[D_X]]
  // CHECK-NEXT: %[[D_Y_EXT:.+]] = arith.extsi %[[D_Y]]
  // CHECK-NEXT: %[[Y_N_EXT:.+]] = arith.extsi %[[SCALE_Y_N]]
  // CHECK-NEXT: %[[X_N_EXT:.+]] = arith.extsi %[[SCALE_X_N]]

  // Compute the bilinear interpolation.

  // CHECK: %[[NDX:.+]] = arith.subi %[[X_N_EXT]], %[[D_X_EXT]]
  // CHECK: %[[WLOLO:.+]] = arith.muli %[[XLOLO]], %[[NDX]]
  // CHECK: %[[WLOHI:.+]] = arith.muli %[[XLOHI]], %[[D_X_EXT]]
  // CHECK: %[[LO:.+]] = arith.addi %[[WLOLO]], %[[WLOHI]]
  // CHECK: %[[NDX:.+]] = arith.subi %[[X_N_EXT]], %[[D_X_EXT]]
  // CHECK: %[[WHILO:.+]] = arith.muli %[[XHILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.muli %[[XHIHI]], %[[D_X_EXT]]
  // CHECK: %[[HI:.+]] = arith.addi %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subi %[[Y_N_EXT]], %[[D_Y_EXT]]
  // CHECK: %[[WLO:.+]] = arith.muli %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.muli %[[HI]], %[[D_Y_EXT]]
  // CHECK: %[[RESULT:.+]] = arith.addi %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]

  // Round to the nearest index.
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 16, 1, 16, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x19x20x1xi8>) -> tensor<1x304x320x1xi48>
  return
}

// -----

// CHECK-LABEL: @resize_nearest_fp32
func.func @resize_nearest_fp32(%input: tensor<1x50x48x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x1600x1536x1xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX0:.+]] = linalg.index 0
  // CHECK: %[[IDX1:.+]] = linalg.index 1
  // CHECK: %[[IDX2:.+]] = linalg.index 2
  // CHECK: %[[IDX3:.+]] = linalg.index 3
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
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
  // CHECK: %[[SCALE_Y_N:.*]] = arith.uitofp %[[ISCALE_Y_N]]
  // CHECK: %[[SCALE_Y_D:.*]] = arith.uitofp %[[ISCALE_Y_D]]
  // CHECK: %[[OFFSET_Y:.*]] = arith.sitofp %[[IOFFSET_Y]]
  // CHECK: %[[VAL_29:.*]] = arith.mulf %[[Y0]], %[[SCALE_Y_D]]
  // CHECK: %[[VAL_31:.*]] = arith.addf %[[VAL_29]], %[[OFFSET_Y]]
  // CHECK: %[[VAL_33:.*]] = arith.divf %[[VAL_31]], %[[SCALE_Y_N]]
  // CHECK: %[[VAL_35:.*]] = math.floor %[[VAL_33]]
  // CHECK: %[[D_Y:.*]] = arith.subf %[[VAL_33]], %[[VAL_35]]
  // CHECK: %[[VAL_39:.*]] = arith.fptosi %[[VAL_35]]

  // CHECK: %[[X0:.+]] = arith.uitofp %[[X]]
  // CHECK: %[[SCALE_X_N:.*]] = arith.uitofp %[[ISCALE_X_N]]
  // CHECK: %[[SCALE_X_D:.*]] = arith.uitofp %[[ISCALE_X_D]]
  // CHECK: %[[OFFSET_X:.*]] = arith.sitofp %[[IOFFSET_X]]
  // CHECK: %[[VAL_30:.*]] = arith.mulf %[[X0]], %[[SCALE_X_D]]
  // CHECK: %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[OFFSET_X]]
  // CHECK: %[[VAL_34:.*]] = arith.divf %[[VAL_32]], %[[SCALE_X_N]]
  // CHECK: %[[VAL_36:.*]] = math.floor %[[VAL_34]]
  // CHECK: %[[D_X:.*]] = arith.subf %[[VAL_34]], %[[VAL_36]]
  // CHECK: %[[VAL_40:.*]] = arith.fptosi %[[VAL_36]]

  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01
  // CHECK: %[[PRED_Y:.*]] = arith.cmpf oge, %[[D_Y]], %[[HALF]]
  // CHECK: %[[ROUND_Y:.*]] = arith.select %[[PRED_Y]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_48:.*]] = arith.addi %[[VAL_39]], %[[ROUND_Y]]
  // CHECK: %[[VAL_50:.*]] = arith.cmpi slt, %[[VAL_48]], %[[ZERO]]
  // CHECK: %[[VAL_51:.*]] = arith.select %[[VAL_50]], %[[ZERO]], %[[VAL_48]]
  // CHECK: %[[VAL_52:.*]] = arith.cmpi slt, %[[YMAX]], %[[VAL_48]]
  // CHECK: %[[VAL_53:.*]] = arith.select %[[VAL_52]], %[[YMAX]], %[[VAL_51]]
  // CHECK: %[[IDY:.*]] = arith.index_cast %[[VAL_53]]

  // CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01
  // CHECK: %[[PRED_X:.*]] = arith.cmpf oge, %[[D_X]], %[[HALF]]
  // CHECK: %[[ROUND_X:.*]] = arith.select %[[PRED_X]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_49:.*]] = arith.addi %[[VAL_40]], %[[ROUND_X]]
  // CHECK: %[[VAL_54:.*]] = arith.cmpi slt, %[[VAL_49]], %[[ZERO]]
  // CHECK: %[[VAL_55:.*]] = arith.select %[[VAL_54]], %[[ZERO]], %[[VAL_49]]
  // CHECK: %[[VAL_56:.*]] = arith.cmpi slt, %[[XMAX]], %[[VAL_49]]
  // CHECK: %[[VAL_57:.*]] = arith.select %[[VAL_56]], %[[XMAX]], %[[VAL_55]]
  // CHECK: %[[IDX:.*]] = arith.index_cast %[[VAL_57]]

  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX0]], %[[IDY]], %[[IDX]], %[[IDX3]]]
  // CHECK: linalg.yield %[[EXTRACT]]

  %output = "tosa.resize"(%input) {mode = "NEAREST_NEIGHBOR", scale = array<i64: 64, 2, 64, 2>, offset = array<i64: -31, -31>, border = array<i64: 31, 31>} : (tensor<1x50x48x1xf32>) -> tensor<1x1600x1536x1xf32>
  return
}

// -----

// CHECK-LABEL: @resize_bilinear_fp
func.func @resize_bilinear_fp(%input: tensor<1x23x24x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x92x96x1xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[Y_MAX:.*]] = arith.constant 22
  // CHECK-DAG: %[[X_MAX:.*]] = arith.constant 23
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
  // CHECK: %[[SCALE_Y_N:.*]] = arith.uitofp %[[ISCALE_Y_N]]
  // CHECK: %[[SCALE_Y_D:.*]] = arith.uitofp %[[ISCALE_Y_D]]
  // CHECK: %[[OFFSET_Y:.*]] = arith.sitofp %[[IOFFSET_Y]]
  // CHECK: %[[VAL_29:.*]] = arith.mulf %[[Y0]], %[[SCALE_Y_D]]
  // CHECK: %[[VAL_31:.*]] = arith.addf %[[VAL_29]], %[[OFFSET_Y]]
  // CHECK: %[[VAL_33:.*]] = arith.divf %[[VAL_31]], %[[SCALE_Y_N]]
  // CHECK: %[[VAL_35:.*]] = math.floor %[[VAL_33]]
  // CHECK: %[[D_Y:.*]] = arith.subf %[[VAL_33]], %[[VAL_35]]
  // CHECK: %[[I_Y:.*]] = arith.fptosi %[[VAL_35]]

  // CHECK: %[[X0:.+]] = arith.uitofp %[[X]]
  // CHECK: %[[SCALE_X_N:.*]] = arith.uitofp %[[ISCALE_X_N]]
  // CHECK: %[[SCALE_X_D:.*]] = arith.uitofp %[[ISCALE_X_D]]
  // CHECK: %[[OFFSET_X:.*]] = arith.sitofp %[[IOFFSET_X]]
  // CHECK: %[[VAL_30:.*]] = arith.mulf %[[X0]], %[[SCALE_X_D]]
  // CHECK: %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[OFFSET_X]]
  // CHECK: %[[VAL_34:.*]] = arith.divf %[[VAL_32]], %[[SCALE_X_N]]
  // CHECK: %[[VAL_36:.*]] = math.floor %[[VAL_34]]
  // CHECK: %[[D_X:.*]] = arith.subf %[[VAL_34]], %[[VAL_36]]
  // CHECK: %[[I_X:.*]] = arith.fptosi %[[VAL_36]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK: %[[ONE:.*]] = arith.constant 1

  // Bound check each dimension.

  // CHECK: %[[Y1:.*]] = arith.addi %[[I_Y]], %[[ONE]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_Y]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[I_Y]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[I_Y]]
  // CHECK: %[[YLO:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y1]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[Y1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[Y1]]
  // CHECK: %[[YHI:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]
  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]

  // CHECK: %[[X1:.*]] = arith.addi %[[I_X]], %[[ONE]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_X]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[I_X]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[I_X]]
  // CHECK: %[[XLO:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X1]], %[[ZERO]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[ZERO]], %[[X1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[X1]]
  // CHECK: %[[XHI:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

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
  // CHECK: %[[NDX:.+]] = arith.subf %[[ONE]], %[[D_X]]
  // CHECK: %[[WHILO:.+]] = arith.mulf %[[HILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.mulf %[[HIHI]], %[[D_X]]
  // CHECK: %[[HI:.+]] = arith.addf %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subf %[[ONE]], %[[D_Y]]
  // CHECK: %[[WLO:.+]] = arith.mulf %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.mulf %[[HI]], %[[D_Y]]
  // CHECK: %[[RESULT:.+]] = arith.addf %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]

  // Round by bilinear interpolation
  %output = "tosa.resize"(%input) {mode = "BILINEAR", scale = array<i64: 4, 1, 4, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x23x24x1xf32>) -> tensor<1x92x96x1xf32>

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
  %output = "tosa.resize"(%input) { scale = array<i64: 4, 2, 4, 2>, offset = array<i64: -1, -1>, border = array<i64: 1, 1>, mode = "BILINEAR" } : (tensor<?x2x2x1xi8>)  -> (tensor<?x4x4x1xi32>)
  return
}

// -----

// CHECK-LABEL: @resize_bilinear_int48
func.func @resize_bilinear_int48(%arg0: tensor<1x19x19x1xi16>) {
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 16, 1, 16, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x19x19x1xi16>) -> tensor<1x289x289x1xi48>
           return
}

// -----

// CHECK-LABEL: skip_interpolate_bilinear_i8
func.func @skip_interpolate_bilinear_i8(%arg0 : tensor<3x1x2x7xi8>) -> tensor<3x1x5x7xi32> {
  // CHECK:  %[[GENERIC:.+]] = linalg.generic
  // CHECK:    %[[BATCH:.+]] = linalg.index 0
  // CHECK:    %[[CHANNEL:.+]] = linalg.index 3
  // CHECK-DAG:    %[[C3:.+]] = arith.constant 3
  // CHECK-DAG:    %[[C2:.+]] = arith.constant 2
  // CHECK:    %[[EXTRACT0:.+]] = tensor.extract %arg0[%[[BATCH]], %{{.+}}, %{{.+}}, %[[CHANNEL]]] : tensor<3x1x2x7xi8>
  // CHECK:    %[[EXTRACT1:.+]] = tensor.extract %arg0[%[[BATCH]], %{{.+}}, %{{.+}}, %[[CHANNEL]]] : tensor<3x1x2x7xi8>
  // CHECK:    %[[EXT0:.+]] = arith.extsi %[[EXTRACT0]] : i8 to i32
  // CHECK:    %[[EXT1:.+]] = arith.extsi %[[EXTRACT1]] : i8 to i32
  // CHECK:    %[[SUB:.+]] = arith.subi %[[C3]], %[[DX:.+]]
  // CHECK:    %[[MUL0:.+]] = arith.muli %[[EXT0]], %[[SUB]]
  // CHECK:    %[[MUL1:.+]] = arith.muli %[[EXT1]], %[[DX]]
  // CHECK:    %[[ADD:.+]] = arith.addi %[[MUL0]], %[[MUL1]]
  // CHECK:    %[[RES:.+]] = arith.muli %[[ADD]], %[[C2]]
  // CHECK:    linalg.yield %[[RES]]
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 2, 1, 3, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x2x7xi8>) -> tensor<3x1x5x7xi32>

  // CHECK:  return %[[GENERIC]]
  return %resize : tensor<3x1x5x7xi32>
}

// CHECK-LABEL: skip_interpolate_bilinear_f32
func.func @skip_interpolate_bilinear_f32(%arg0 : tensor<3x1x2x7xf32>) -> tensor<3x1x5x7xf32> {
  // CHECK:  %[[GENERIC:.+]] = linalg.generic
  // CHECK:    %[[BATCH:.+]] = linalg.index 0 : index
  // CHECK:    %[[CHANNEL:.+]] = linalg.index 3 : index
  // CHECK:    %[[EXTRACT0:.+]] = tensor.extract %arg0[%[[BATCH]], %{{.+}}, %{{.+}}, %[[CHANNEL]]] : tensor<3x1x2x7xf32>
  // CHECK:    %[[EXTRACT1:.+]] = tensor.extract %arg0[%[[BATCH]], %{{.+}}, %{{.+}}, %[[CHANNEL]]] : tensor<3x1x2x7xf32>
  // CHECK:    %[[C1:.+]] = arith.constant 1.000000e+00
  // CHECK:    %[[SUB:.+]] = arith.subf %[[C1]], %[[DX:.+]]
  // CHECK:    %[[MUL0:.+]] = arith.mulf %[[EXTRACT0]], %[[SUB]]
  // CHECK:    %[[MUL1:.+]] = arith.mulf %[[EXTRACT1]], %[[DX]]
  // CHECK:    %[[ADD:.+]] = arith.addf %[[MUL0]], %[[MUL1]]
  // CHECK:    linalg.yield %[[ADD]]
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = array<i64: 2, 1, 3, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<3x1x2x7xf32>) -> tensor<3x1x5x7xf32>

  // CHECK:  return %[[GENERIC]]
  return %resize : tensor<3x1x5x7xf32>
}
