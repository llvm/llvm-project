// RUN: mlir-opt --split-input-file -pass-pipeline="func.func(tosa-to-linalg-named)" %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: @matmul
func.func @matmul(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x6xf32>) -> (tensor<1x5x6xf32>) {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : f32) outs([[INIT]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x6xf32>) outs([[FILLED]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x3xf32>, tensor<1x3x6xf32>)  -> (tensor<1x5x6xf32>)
  return %0 : tensor<1x5x6xf32>
}

// -----


// CHECK-LABEL: @matmul_quantized
func.func @matmul_quantized(%arg0: tensor<1x5x3xi8>, %arg1: tensor<1x3x6xi8>) -> (tensor<1x5x6xi32>) {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[FILLED:%.+]] = linalg.fill ins([[C0]] : i32) outs([[INIT]] : tensor<1x5x6xi32>) -> tensor<1x5x6xi32>
  // CHECK: [[ONE:%.+]] = arith.constant 1
  // CHECK: [[TWO:%.+]] = arith.constant 2
  // CHECK: linalg.quantized_batch_matmul ins(%arg0, %arg1, [[ONE]], [[TWO]] : tensor<1x5x3xi8>, tensor<1x3x6xi8>, i32, i32) outs([[FILLED]] : tensor<1x5x6xi32>) -> tensor<1x5x6xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) {quantization_info = #tosa.matmul_quant<a_zp = 1, b_zp = 2>} : (tensor<1x5x3xi8>, tensor<1x3x6xi8>) -> (tensor<1x5x6xi32>)
  return %0 : tensor<1x5x6xi32>
}

// -----

// CHECK-LABEL: @matmul_dyn_batch
func.func @matmul_dyn_batch(%arg0: tensor<?x5x3xf32>, %arg1: tensor<?x3x6xf32>) -> (tensor<?x5x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[C0_0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM]])
  // CHECK: %[[FILLED:.+]] = linalg.fill ins(%[[C0_0]] : f32) outs(%[[INIT]] : tensor<?x5x6xf32>) -> tensor<?x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x5x3xf32>, tensor<?x3x6xf32>) outs(%[[FILLED]] : tensor<?x5x6xf32>) -> tensor<?x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x5x3xf32>, tensor<?x3x6xf32>)  -> (tensor<?x5x6xf32>)
  return %0 : tensor<?x5x6xf32>
}

// -----

// CHECK-LABEL: @matmul_dyn_independent_dim
func.func @matmul_dyn_independent_dim(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x?xf32>) -> (tensor<1x5x?xf32>) {
  // CHECK: %[[C2:.+]] = arith.constant 2
  // CHECK: %[[DIM:.+]] = tensor.dim %arg1, %[[C2]]
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM]])
  // CHECK: %[[FILLED:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[INIT]] : tensor<1x5x?xf32>) -> tensor<1x5x?xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x?xf32>) outs(%[[FILLED]] : tensor<1x5x?xf32>) -> tensor<1x5x?xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x3xf32>, tensor<1x3x?xf32>)  -> (tensor<1x5x?xf32>)
  return %0 : tensor<1x5x?xf32>
}

// -----

// CHECK-LABEL: @matmul_dyn_independent_dim
func.func @matmul_dyn_independent_dim(%arg0: tensor<1x5x?xf32>, %arg1: tensor<1x?x6xf32>) -> (tensor<1x5x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[FILLED:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[INIT]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x?xf32>, tensor<1x?x6xf32>) outs(%[[FILLED]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x?xf32>, tensor<1x?x6xf32>)  -> (tensor<1x5x6xf32>)
  return %0 : tensor<1x5x6xf32>
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @fully_connected
func.func @fully_connected(%arg0: tensor<5x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> (tensor<5x6xf32>) {
  // CHECK: [[INITT:%.+]] = tensor.empty()
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[ZERO]]{{.*}}outs([[INITT]]
  // CHECK: [[PERM:%.+]] = arith.constant dense<[1, 0]>
  // CHECK: [[TRANSPOSE:%.+]] = "tosa.transpose"(%arg1, [[PERM]])
  // CHECK: [[INITB:%.+]] = tensor.empty()
  // CHECK: [[MATMUL:%.+]] = linalg.matmul ins(%arg0, [[TRANSPOSE]] : tensor<5x3xf32>, tensor<3x6xf32>) outs([[FILL]] : tensor<5x6xf32>) -> tensor<5x6xf32>
  // CHECK: [[ADDED:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, [[MATMUL]] : tensor<6xf32>, tensor<5x6xf32>) outs([[INITB]] : tensor<5x6xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32

  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<5x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<5x6xf32>)
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @quantized_fully_connected
func.func @quantized_fully_connected(%arg0: tensor<5x3xi8>, %arg1: tensor<6x3xi8>, %arg2: tensor<6xi32>) -> (tensor<5x6xi32>) {
  // CHECK: [[INITT:%.+]] = tensor.empty()
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[ZERO]]{{.*}}outs([[INITT]]
  // CHECK: [[PERM:%.+]] = arith.constant dense<[1, 0]>
  // CHECK: [[TRANSPOSE:%.+]] = "tosa.transpose"(%arg1, [[PERM]])
  // CHECK: [[INITB:%.+]] = tensor.empty()
  // CHECK: [[ONE:%.+]] = arith.constant 1
  // CHECK: [[TWO:%.+]] = arith.constant 2
  // CHECK: [[MATMUL:%.+]] = linalg.quantized_matmul ins(%arg0, [[TRANSPOSE]], [[ONE]], [[TWO]] : tensor<5x3xi8>, tensor<3x6xi8>, i32, i32) outs([[FILL]] : tensor<5x6xi32>) -> tensor<5x6xi32>
  // CHECK: [[ADDED:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, [[MATMUL]] : tensor<6xi32>, tensor<5x6xi32>) outs([[INITB]]
  // CHECK: ^bb0([[IN1:%.+]]: i32, [[IN2:%.+]]: i32, [[UNUSED:%.+]]: i32):
  // CHECK:   [[ADD:%.+]] = arith.addi
  // CHECK:   linalg.yield [[ADD]] : i32
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) {quantization_info = #tosa.conv_quant<input_zp = 1, weight_zp = 2>} : (tensor<5x3xi8>, tensor<6x3xi8>, tensor<6xi32>)  -> (tensor<5x6xi32>)
  return %0 : tensor<5x6xi32>
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @fully_connected_dyn
func.func @fully_connected_dyn(%arg0: tensor<?x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> (tensor<?x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INITT:.+]] = tensor.empty(%[[DIM]])
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INITT]]
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 0]>
  // CHECK: %[[TRANSPOSE:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: %[[INITB:.+]] = tensor.empty(%[[DIM]])
  // CHECK: %[[MATMUL:.+]] = linalg.matmul ins(%arg0, %[[TRANSPOSE]] : tensor<?x3xf32>, tensor<3x6xf32>) outs(%[[FILL]] : tensor<?x6xf32>) -> tensor<?x6xf32>
  // CHECK: %[[ADDED:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, %[[MATMUL]] : tensor<6xf32>, tensor<?x6xf32>) outs(%[[INITB]] : tensor<?x6xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   %[[ADD:.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield %[[ADD]] : f32

  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<?x6xf32>)
  return %0 : tensor<?x6xf32>
}

// -----

// CHECK-LABEL: @max_pool
func.func @max_pool(%arg0: tensor<1x6x34x62xf32>) -> () {
  // CHECK-DAG: [[CONST:%.+]] = arith.constant -3.40282347E+38
  // CHECK-DAG: [[INIT:%.+]] = tensor.empty()
  // CHECK-DAG: [[FILL:%.+]] = linalg.fill ins([[CONST]]{{.*}}outs([[INIT]]
  // CHECK-DAG: [[KERNEL:%.+]] = tensor.empty()
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, [[KERNEL]] : tensor<1x6x34x62xf32>, tensor<3x3xf32>) outs([[FILL]] : tensor<1x4x32x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x4x32x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_padded
func.func @max_pool_padded(%arg0: tensor<1x6x34x62xf32>) -> () {
  // CHECK-DAG: [[CONST:%.+]] = arith.constant -3.40282347E+38 : f32
  // CHECK-DAG: [[PAD:%.+]] = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 0, 1, 0]
  // CHECK-DAG:   tensor.yield [[CONST]]
  // CHECK-DAG: [[INITVAL:%.+]] = arith.constant -3.40282347E+38 : f32
  // CHECK-DAG: [[INIT:%.+]] = tensor.empty()
  // CHECK-DAG: [[FILL:%.+]] = linalg.fill ins([[INITVAL]]{{.*}}outs([[INIT]]
  // CHECK-DAG: [[KERNEL:%.+]] = tensor.empty()
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins([[PAD]], [[KERNEL]] : tensor<1x6x35x62xf32>, tensor<3x3xf32>) outs([[FILL]] : tensor<1x4x33x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 1], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x4x33x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_dyn
func.func @max_pool_dyn(%arg0: tensor<?x6x34x62xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[CONST:.+]] = arith.constant -3.40282347E+38
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CONST]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[KERNEL:.+]] = tensor.empty()
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %[[KERNEL]] : tensor<?x6x34x62xf32>, tensor<3x3xf32>) outs(%[[FILL]] : tensor<?x4x32x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<?x6x34x62xf32>)  -> (tensor<?x4x32x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_i8
func.func @max_pool_i8(%arg0: tensor<1x6x34x62xi8>) -> () {
  // CHECK: arith.constant -128
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi8>)  -> (tensor<1x4x32x62xi8>)
  return
}

// CHECK-LABEL: @max_pool_i16
func.func @max_pool_i16(%arg0: tensor<1x6x34x62xi16>) -> () {
  // CHECK: arith.constant -32768
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi16>)  -> (tensor<1x4x32x62xi16>)
  return
}

// CHECK-LABEL: @max_pool_i32
func.func @max_pool_i32(%arg0: tensor<1x6x34x62xi32>) -> () {
  // CHECK: arith.constant -2147483648
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi32>)  -> (tensor<1x4x32x62xi32>)
  return
}
// -----

// CHECK-LABEL: @avg_pool
func.func @avg_pool(%arg0: tensor<1x6x34x62xf32>) -> (tensor<1x5x33x62xf32>) {
  // Initial piece computes the sum of the pooling region, with appropriate padding.
  // CHECK: [[CONST:%.+]] = arith.constant 0
  // CHECK: [[PAD:%.+]] = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK: [[CONST:%.+]] = arith.constant 0
  // CHECK: [[POOLINIT:%.+]] = tensor.empty()
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CONST]]{{.*}}outs([[POOLINIT]]
  // CHECK: [[KERNEL:%.+]] = tensor.empty()
  // CHECK: [[POOL:%.+]] = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins([[PAD]], [[KERNEL]] : tensor<1x8x36x62xf32>, tensor<4x4xf32>) outs([[FILL]] : tensor<1x5x33x62xf32>)
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[POOL]] : tensor<1x5x33x62xf32>) outs([[INIT]] : tensor<1x5x33x62xf32>)
  // CHECK:   [[ZERO:%.0]] = arith.constant 0
  // CHECK:   [[ONE:%.+]] = arith.constant 1
  // CHECK:   [[HEIGHT:%.+]] = arith.constant 4
  // CHECK:   [[WIDTH:%.+]] = arith.constant 32
  // CHECK:   [[IDX1:%.+]] = linalg.index 1
  // CHECK:   [[IDX2:%.+]] = linalg.index 2

  // The large block below computes what portion of the kernel is within non-padded input.
  // CHECK:   [[NY:%.+]] = arith.subi [[HEIGHT]], [[IDX1]]
  // CHECK:   [[NX:%.+]] = arith.subi [[WIDTH]], [[IDX2]]
  // CHECK:   [[KH:%.+]] = arith.constant 4
  // CHECK:   [[PAD0:%.+]] = arith.constant 1
  // CHECK:   [[SUBP0:%.+]] = arith.subi [[IDX1]], [[PAD0]]
  // CHECK:   [[P0CMP:%.+]] = arith.cmpi slt, [[SUBP0]], [[ZERO]]
  // CHECK:   [[SELP0:%.+]] = arith.select [[P0CMP]], [[SUBP0]], [[ZERO]]
  // CHECK:   [[ADDP0:%.+]] = arith.addi [[KH]], [[SELP0]]
  // CHECK:   [[PAD1:%.+]] = arith.constant 1
  // CHECK:   [[SUBP1:%.+]] = arith.subi [[NY]], [[PAD1]]
  // CHECK:   [[P1CMP:%.+]] = arith.cmpi slt, [[SUBP1]], [[ZERO]]
  // CHECK:   [[SELP1:%.+]] = arith.select [[P1CMP]], [[SUBP1]], [[ZERO]]
  // CHECK:   [[ADDP1:%.+]] = arith.addi [[ADDP0]], [[SELP1]]
  // CHECK:   [[YCMP:%.+]] = arith.cmpi slt, [[ADDP1]], [[ONE]]
  // CHECK:   [[YSEL:%.+]] = arith.select [[YCMP]], [[ONE]], [[ADDP1]]
  // CHECK:   [[KW:%.+]] = arith.constant 4 : index
  // CHECK:   [[PAD2:%.+]] = arith.constant 1 : index
  // CHECK:   [[SUBP2:%.+]] = arith.subi [[IDX2]], [[PAD2]]
  // CHECK:   [[P2CMP:%.+]] = arith.cmpi slt, [[SUBP2]], [[ZERO]]
  // CHECK:   [[SELP2:%.+]] = arith.select [[P2CMP]], [[SUBP2]], [[ZERO]]
  // CHECK:   [[ADDP2:%.+]] = arith.addi [[KW]], [[SELP2]]
  // CHECK:   [[PAD3:%.+]] = arith.constant 1 : index
  // CHECK:   [[SUBP3:%.+]] = arith.subi [[NX]], [[PAD3]]
  // CHECK:   [[P3CMP:%.+]] = arith.cmpi slt, [[SUBP3]], [[ZERO]]
  // CHECK:   [[SELP3:%.+]] = arith.select [[P3CMP]], [[SUBP3]], [[ZERO]]
  // CHECK:   [[ADDP3:%.+]] = arith.addi [[ADDP2]], [[SELP3]]
  // CHECK:   [[XCMP:%.+]] = arith.cmpi slt, [[ADDP3]], [[ONE]]
  // CHECK:   [[XSEL:%.+]] = arith.select [[XCMP]], [[ONE]], [[ADDP3]]

  // Given the valid coverage of the pooling region, normalize the summation.
  // CHECK:   [[C:%.+]] = arith.muli [[YSEL]], [[XSEL]]
  // CHECK:   [[CI:%.+]] = arith.index_cast [[C]]
  // CHECK:   [[CF:%.+]] = arith.sitofp [[CI]]
  // CHECK:   [[RESULT:%.+]] = arith.divf %arg1, [[CF]]
  // CHECK:   linalg.yield [[RESULT]]
  %0 = "tosa.avg_pool2d"(%arg0) {pad = [1, 1, 1, 1], kernel = [4, 4], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x5x33x62xf32>)
  return %0 : tensor<1x5x33x62xf32>
}

// -----

// CHECK-LABEL: @avg_pool_dyn
func.func @avg_pool_dyn(%arg0: tensor<?x6x34x62xf32>) -> (tensor<?x5x33x62xf32>) {
  // The calculations remain the same as above, only testing for dyn behavior
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[PAD:.+]] = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK: %[[POOLINIT:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[KERNEL:.+]] = tensor.empty()
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%[[PAD]], %[[KERNEL]] : tensor<?x8x36x62xf32>, tensor<4x4xf32>) outs(%[[FILL]] : tensor<?x5x33x62xf32>)
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[POOL]] : tensor<?x5x33x62xf32>) outs(%[[INIT]] : tensor<?x5x33x62xf32>)
  %0 = "tosa.avg_pool2d"(%arg0) {pad = [1, 1, 1, 1], kernel = [4, 4], stride = [1, 1]} : (tensor<?x6x34x62xf32>)  -> (tensor<?x5x33x62xf32>)
  return %0 : tensor<?x5x33x62xf32>
}

// -----

// CHECK-LABEL: @avg_pool_i8
func.func @avg_pool_i8(%arg0 : tensor<1x128x128x2xi8>) -> () {

  // CHECK: linalg.pooling_nhwc_sum
  // CHECK: linalg.generic

  // CHECK: %[[INZP:.+]] = arith.constant -128
  // CHECK: %[[INZP_OFF:.+]] = arith.muli %{{.+}}, %[[INZP]]
  // CHECK: %[[OFFSETED:.+]] = arith.subi %arg1, %[[INZP_OFF]]
  // CHECK: %[[NUMERATOR:.+]] = arith.constant 1073741825
  // CHECK: %[[MULTIPLIER:.+]] = arith.divui %[[NUMERATOR]], %{{.+}}
  // CHECK: %[[SHIFT:.+]] = arith.constant 30
  // CHECK: %[[SCALE:.+]] = "tosa.apply_scale"(%{{.+}}, %[[MULTIPLIER]], %[[SHIFT]]) {double_round = false}
  // CHECK: %[[OUTZP:.+]] = arith.constant -128
  // CHECK: %[[OUT:.+]] = arith.addi %[[SCALE]], %[[OUTZP]]
  // CHECK: %[[MIN:.+]] = arith.constant -128
  // CHECK: %[[MAX:.+]] = arith.constant 127
  // CHECK: %[[CMP_MIN:.+]] = arith.cmpi slt, %[[OUT]], %[[MIN]]
  // CHECK: %[[CLMP_MIN:.+]] = arith.select %[[CMP_MIN]], %[[MIN]], %[[OUT]]
  // CHECK: %[[CMP_MAX:.+]] = arith.cmpi slt, %[[MAX]], %[[OUT]]
  // CHECK: %[[CLMP_MAX:.+]] = arith.select %[[CMP_MAX]], %[[MAX]], %[[CLMP_MIN]]
  // CHECK: %[[TRUNC:.+]] = arith.trunci %[[CLMP_MAX]]
  // CHECK: linalg.yield %[[TRUNC]]
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 4], pad = [0, 0, 0, 0], quantization_info = #tosa.unary_quant<input_zp = -128, output_zp = -128>, stride = [4, 4]} : (tensor<1x128x128x2xi8>) -> tensor<1x32x32x2xi8>
  return
}

// -----

// CHECK-LABEL: @avg_pool_i16
func.func @avg_pool_i16(%arg0 : tensor<1x128x128x2xi16>) -> () {

  // CHECK: linalg.pooling_nhwc_sum
  // CHECK: linalg.generic

  // CHECK: %[[INZP:.+]] = arith.constant -128
  // CHECK: %[[INZP_OFF:.+]] = arith.muli %{{.+}}, %[[INZP]]
  // CHECK: %[[OFFSETED:.+]] = arith.subi %arg1, %[[INZP_OFF]]
  // CHECK: %[[NUMERATOR:.+]] = arith.constant 1073741825
  // CHECK: %[[MULTIPLIER:.+]] = arith.divui %[[NUMERATOR]], %{{.+}}
  // CHECK: %[[SHIFT:.+]] = arith.constant 30
  // CHECK: %[[SCALE:.+]] = "tosa.apply_scale"(%{{.+}}, %[[MULTIPLIER]], %[[SHIFT]]) {double_round = false}
  // CHECK: %[[OUTZP:.+]] = arith.constant -128
  // CHECK: %[[OUT:.+]] = arith.addi %[[SCALE]], %[[OUTZP]]
  // CHECK: %[[MIN:.+]] = arith.constant -32768
  // CHECK: %[[MAX:.+]] = arith.constant 32767
  // CHECK: %[[CMP_MIN:.+]] = arith.cmpi slt, %[[OUT]], %[[MIN]]
  // CHECK: %[[CLMP_MIN:.+]] = arith.select %[[CMP_MIN]], %[[MIN]], %[[OUT]]
  // CHECK: %[[CMP_MAX:.+]] = arith.cmpi slt, %[[MAX]], %[[OUT]]
  // CHECK: %[[CLMP_MAX:.+]] = arith.select %[[CMP_MAX]], %[[MAX]], %[[CLMP_MIN]]
  // CHECK: %[[TRUNC:.+]] = arith.trunci %[[CLMP_MAX]]
  // CHECK: linalg.yield %[[TRUNC]]
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 4], pad = [0, 0, 0, 0], quantization_info = #tosa.unary_quant<input_zp = -128, output_zp = -128>, stride = [4, 4]} : (tensor<1x128x128x2xi16>) -> tensor<1x32x32x2xi16>
  return
}

// -----

// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @conv2d_f32
func.func @conv2d_f32(%input: tensor<1x49x42x27xf32>, %weights: tensor<28x3x3x27xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 2, 3, 0]>
  // CHECK: %[[W:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: %[[M_IN:.+]] = tensor.empty()
  // CHECK: %[[CST:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[B_IN:.+]] = tensor.empty()
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %[[W]] : tensor<1x49x42x27xf32>, tensor<3x3x27x28xf32>) outs(%[[FILL]] : tensor<1x45x40x28xf32>)
  // CHECK: %[[B:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[CONV]] : tensor<28xf32>, tensor<1x45x40x28xf32>) outs(%[[B_IN]] : tensor<1x45x40x28xf32>)
  // CHECK:   arith.addf
  // CHECK:   linalg.yield
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [2, 1]} : (tensor<1x49x42x27xf32>, tensor<28x3x3x27xf32>, tensor<28xf32>)  -> (tensor<1x45x40x28xf32>)
  return
}

// -----

// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @conv2d_dyn
func.func @conv2d_dyn(%input: tensor<?x49x42x27xf32>, %weights: tensor<28x3x3x27xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 2, 3, 0]>
  // CHECK: %[[W:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: %[[M_IN:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[CST:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[B_IN:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %[[W]] : tensor<?x49x42x27xf32>, tensor<3x3x27x28xf32>) outs(%[[FILL]] : tensor<?x45x40x28xf32>)
  // CHECK: %[[B:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[CONV]] : tensor<28xf32>, tensor<?x45x40x28xf32>) outs(%[[B_IN]] : tensor<?x45x40x28xf32>)
  // CHECK:   %[[ADD:.+]] = arith.addf
  // CHECK:   linalg.yield %[[ADD]] : f32
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [2, 1]} : (tensor<?x49x42x27xf32>, tensor<28x3x3x27xf32>, tensor<28xf32>)  -> (tensor<?x45x40x28xf32>)
  return
}

// -----

// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @conv2d_dyn_w_h
func.func @conv2d_dyn_w_h(%input: tensor<1x?x?x27xf32>, %weights: tensor<28x3x3x27xf32>, %bias: tensor<28xf32>) -> () {
  // Computing output height
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[H:.+]] = tensor.dim %arg0, %[[C1]]
  // CHECK: %[[C1_0:.+]] = arith.constant 1
  // CHECK: %[[KH:.+]] = tensor.dim %arg1, %[[C1_0]]
  // CHECK: %[[ONE:.+]] = arith.constant 1 : index
  // CHECK: %[[PAD_0:.+]] = arith.constant 0 : index
  // CHECK: %[[ADD_PAD_0:.+]] = arith.addi %[[H]], %[[PAD_0]] : index
  // CHECK: %[[PAD_1:.+]] = arith.constant 0 : index
  // CHECK: %[[ADD_PAD_1:.+]] = arith.addi %[[ADD_PAD_0]], %[[PAD_1]] : index
  // CHECK: %[[SUB_ONE:.+]] = arith.subi %[[KH]], %[[ONE]] : index
  // CHECK: %[[DIL_H:.+]] = arith.constant 2 : index
  // CHECK: %[[DILATED:.+]] = arith.muli %[[DIL_H]], %[[SUB_ONE]] : index
  // CHECK: %[[ADD_ONE:.+]] = arith.addi %[[DILATED]], %[[ONE]] : index
  // CHECK: %[[SUBTRACTED:.+]] = arith.subi %[[ADD_PAD_1]], %[[ADD_ONE]] : index
  // CHECK: %[[STRIDE_H:.+]] = arith.constant 1 : index
  // CHECK: %[[DIVIDED:.+]] = arith.divui %[[SUBTRACTED]], %[[STRIDE_H]] : index
  // CHECK: %[[H_OUT:.+]] = arith.addi %[[DIVIDED]], %[[ONE]] : index

  // Computing output width
  // CHECK: %[[C2:.+]] = arith.constant 2
  // CHECK: %[[W:.+]] = tensor.dim %arg0, %[[C2]]
  // CHECK: %[[C2_0:.+]] = arith.constant 2
  // CHECK: %[[KW:.+]] = tensor.dim %arg1, %[[C2_0]]
  // CHECK: %[[ONE_0:.+]] = arith.constant 1 : index
  // CHECK: %[[PAD_2:.+]] = arith.constant 0 : index
  // CHECK: %[[ADD_PAD_2:.+]] = arith.addi %[[W]], %[[PAD_2]] : index
  // CHECK: %[[PAD_3:.+]] = arith.constant 0 : index
  // CHECK: %[[ADD_PAD_3:.+]] = arith.addi %[[ADD_PAD_2]], %[[PAD_3]] : index
  // CHECK: %[[SUB_ONE_0:.+]] = arith.subi %[[KW]], %[[ONE_0]] : index
  // CHECK: %[[DIL_W:.+]] = arith.constant 1 : index
  // CHECK: %[[DILATED_0:.+]] = arith.muli %[[DIL_W]], %[[SUB_ONE_0]] : index
  // CHECK: %[[ADD_ONE_0:.+]] = arith.addi %[[DILATED_0]], %[[ONE_0]] : index
  // CHECK: %[[SUBTRACTED_0:.+]] = arith.subi %[[ADD_PAD_3]], %[[ADD_ONE_0]] : index
  // CHECK: %[[STRIDE_W:.+]] = arith.constant 1 : index
  // CHECK: %[[DIVIDED_0:.+]] = arith.divui %[[SUBTRACTED_0]], %[[STRIDE_W]] : index
  // CHECK: %[[W_OUT:.+]] = arith.addi %[[DIVIDED_0]], %[[ONE_0]] : index

  // Running convolution
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 2, 3, 0]>
  // CHECK: %[[WEIGHT:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: %[[M_IN:.+]] = tensor.empty(%[[H_OUT]], %[[W_OUT]])
  // CHECK: %[[CST:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[B_IN:.+]] = tensor.empty(%[[H_OUT]], %[[W_OUT]])
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %[[WEIGHT]] : tensor<1x?x?x27xf32>, tensor<3x3x27x28xf32>) outs(%[[FILL]] : tensor<1x?x?x28xf32>)
  // CHECK: %[[B:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[CONV]] : tensor<28xf32>, tensor<1x?x?x28xf32>) outs(%[[B_IN]] : tensor<1x?x?x28xf32>)
  // CHECK:   %[[ADD:.+]] = arith.addf
  // CHECK:   linalg.yield %[[ADD]] : f32
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [2, 1]} : (tensor<1x?x?x27xf32>, tensor<28x3x3x27xf32>, tensor<28xf32>)  -> (tensor<1x?x?x28xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_padded_f32
func.func @conv2d_padded_f32(%input: tensor<1x47x40x28xf32>, %weights: tensor<28x3x3x28xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   tensor.yield %[[C0]]
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [2, 1]} : (tensor<1x47x40x28xf32>, tensor<28x3x3x28xf32>, tensor<28xf32>)  -> (tensor<1x45x40x28xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_quant
func.func @conv2d_quant(%arg0 : tensor<1x12x12x1xi8>, %arg1 : tensor<1024x3x3x1xi8>, %arg2 : tensor<1024xi32>) -> () {
  // CHECK:   %[[C22:.+]] = arith.constant -22
  // CHECK: tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   tensor.yield %[[C22]]
  // CHECK: linalg.conv_2d_nhwc_hwcf_q
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [1, 1, 1, 1], quantization_info = #tosa.conv_quant<input_zp = -22, weight_zp = 42>, stride = [1, 1]} : (tensor<1x12x12x1xi8>, tensor<1024x3x3x1xi8>, tensor<1024xi32>) -> tensor<1x12x12x1024xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv
func.func @depthwise_conv(%arg0 : tensor<1x7x5x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[OUT:%.+]] = tensor.empty()
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x7x5x3xf32>, tensor<3x1x3x11xf32>) outs([[FILL]] : tensor<1x5x5x3x11xf32>)
  // CHECK: [[COLLAPSED:%.+]] = tensor.collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<33xf32>, tensor<1x5x5x33xf32>) outs([[OUT]] : tensor<1x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32
  // CHECK: } -> tensor<1x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<1x7x5x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<1x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_dyn
func.func @depthwise_conv_dyn(%arg0 : tensor<?x7x5x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[CST0:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[OUT:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[DEPTH:.+]] = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x7x5x3xf32>, tensor<3x1x3x11xf32>) outs(%[[FILL]] : tensor<?x5x5x3x11xf32>)
  // CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: %[[BIAS:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[COLLAPSED]] : tensor<33xf32>, tensor<?x5x5x33xf32>) outs(%[[OUT]] : tensor<?x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   %[[ADD:.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield %[[ADD]] : f32
  // CHECK: } -> tensor<?x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<?x7x5x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<?x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_strides
func.func @depthwise_conv_strides(%arg0 : tensor<1x11x9x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[OUT:%.+]] = tensor.empty()
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>) outs([[FILL]] : tensor<1x5x5x3x11xf32>)
  // CHECK: [[COLLAPSED:%.+]] = tensor.collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<33xf32>, tensor<1x5x5x33xf32>) outs([[OUT]] : tensor<1x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32
  // CHECK: } -> tensor<1x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1] } : (tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<1x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_quant
func.func @depthwise_conv_quant(%arg0 : tensor<1x12x12x4xi8>, %arg1 : tensor<3x3x4x128xi8>, %arg2 : tensor<512xi32>) -> () {
  // CHECK: [[PADV:%.+]] = arith.constant -128
  // CHECK: [[PAD:%.+]] = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   tensor.yield [[PADV]]

  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[OUT:%.+]] = tensor.empty()
  // CHECK: [[C128:%.+]] = arith.constant -128
  // CHECK: [[C42:%.+]] = arith.constant 42
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins([[PAD]], %arg1, [[C128]], [[C42]] : tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, i32, i32) outs([[FILL]] : tensor<1x12x12x4x128xi32>)
  // CHECK: [[COLLAPSED:%.+]] = tensor.collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<512xi32>, tensor<1x12x12x512xi32>) outs([[OUT]] : tensor<1x12x12x512xi32>) {
  // CHECK: ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  
  // CHECK:   [[ADD:%.+]] = arith.addi %arg3, %arg4 : i32
  // CHECK:   linalg.yield [[ADD]] : i32
  // CHECK: } -> tensor<1x12x12x512xi32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [1, 1, 1, 1], quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 42>, stride = [1, 1], dilation = [1, 1] } : (tensor<1x12x12x4xi8>, tensor<3x3x4x128xi8>, tensor<512xi32>)  -> tensor<1x12x12x512xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_quant_dilations
func.func @depthwise_conv_quant_dilations(%arg0 : tensor<1x14x14x4xi8>, %arg1 : tensor<3x3x4x128xi8>, %arg2 : tensor<512xi32>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[OUT:%.+]] = tensor.empty()
  // CHECK: [[C128:%.+]] = arith.constant -128
  // CHECK: [[C42:%.+]] = arith.constant 42
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, [[C128]], [[C42]] : tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, i32, i32) outs([[FILL]] : tensor<1x10x10x4x128xi32>)
  // CHECK: [[COLLAPSED:%.+]] = tensor.collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<512xi32>, tensor<1x10x10x512xi32>) outs([[OUT]] : tensor<1x10x10x512xi32>) {
  // CHECK: ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  
  // CHECK:   [[ADD:%.+]] = arith.addi %arg3, %arg4 : i32
  // CHECK:   linalg.yield [[ADD]] : i32
  // CHECK: } -> tensor<1x10x10x512xi32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 42>, stride = [1, 1], dilation = [2, 2] } : (tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, tensor<512xi32>)  -> tensor<1x10x10x512xi32>
  return
}

// CHECK-LABEL: @depthwise_conv2d_dyn_w_h
func.func @depthwise_conv2d_dyn_w_h(%arg0: tensor<2x?x?x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: arith.addi
  // CHECK: arith.subi
  // CHECK: arith.muli
  // CHECK: arith.divui
  // CHECK: %[[PADDED:.+]] = tensor.pad %arg0 low[0, 1, 3, 0] high[0, 2, 4, 0] {
  // CHECK: ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
  // CHECK: tensor.yield %cst : f32
  // CHECK:  } : tensor<2x?x?x3xf32> to tensor<2x?x?x3xf32>
  // CHECK: %[[CONV:.+]] = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} ins(%[[PADDED]], %arg1 : tensor<2x?x?x3xf32>, tensor<3x6x3x5xf32>) outs(%22 : tensor<2x?x?x3x5xf32>) -> tensor<2x?x?x3x5xf32>
  // CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[CONV]] {{\[}}[0], [1], [2], [3, 4]]
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [1, 2, 3, 4], dilation = [2, 1], stride = [1, 2]} : (tensor<2x?x?x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x?x?x15xf32>
  return
}
