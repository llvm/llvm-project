// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,canonicalize,linalg-generalize-named-ops,tosa-to-arith,tosa-totensor,canonicalize))" %s -o %t.model.linalg.mlir
// RUN: mlir-opt --split-input-file --canonicalize --linalg-fuse-elementwise-ops --linalg-inline-scalar-operands --linalg-fold-unit-extent-dims --fold-tensor-subset-ops --canonicalize %t.model.linalg.mlir -o %t.model.linalg.opt.mlir
// RUN: mlir-opt --split-input-file --pass-pipeline='builtin.module(one-shot-bufferize{allow-unknown-ops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}, canonicalize)' %t.model_linalg_opt.mlir -o %t.model.buffers.mlir

// RUN: mlir-opt --split-input-file --canonicalize  --buffer-results-to-out-params --buffer-hoisting --buffer-loop-hoisting --promote-buffers-to-stack --fold-memref-alias-ops --canonicalize  --buffer-deallocation-pipeline --canonicalize %t.model_buffers.mlir -o %t.model.buffers.opt.mlir
// RUN: mlir-opt --split-input-file --canonicalize --convert-linalg-to-loops --fold-memref-alias-ops --canonicalize %t.model.buffers.opt.mlir -o %t.model.scf.mlir

// RUN: mlir-opt --split-input-file --canonicalize --convert-linalg-to-loops --canonicalize %t.model.scf.mlir -o %t.model.scf.1.mlir
// RUN: mlir-opt --split-input-file --canonicalize --fold-memref-alias-ops --normalize-memrefs --canonicalize %t.model.scf.1.mlir -o %t.model.scf.2.mlir


// RUN: mlir-opt --split-input-file --arith-expand --canonicalize %t.model.scf.2.mlir -o %t.model.scf.3.mlir
// RUN: mlir-opt --split-input-file --convert-math-to-libm --canonicalize %t.model.scf.4.mlir -o %t.model.scf.5.mlir

// RUN: mlir-opt --split-input-file --convert-func-to-emitc --convert-scf-to-emitc --convert-arith-to-emitc --convert-memref-to-emitc --canonicalize %t.model.scf.5.mlir -o %t.model.emitc.mlir

// RUN: mlir-translate --mlir-to-cpp %t.model.emitc.mlir | FileCheck %s

// CHECK: Fail this test

// -----

func.func @test_abs_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = tosa.abs %arg0 : (tensor<f32>) -> tensor<f32>
	return %0 : tensor<f32>
}

// -----

func.func @test_abs_1d_cast_static_to_dynamic(%arg0: tensor<5xf32>) -> tensor<?xf32> {
  %0 = "tosa.abs"(%arg0) : (tensor<5xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_abs_1d_cast_dynamic_to_static(%arg0: tensor<?xf32>) -> tensor<5xf32> {
  %0 = "tosa.abs"(%arg0) : (tensor<?xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

func.func @test_abs_1d_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tosa.abs %arg0 : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_add_0d(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @test_add_1d_all_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_add_1d_broadcast_dynamic_to_static(%arg0: tensor<5xf32>, %arg1: tensor<?xf32>) -> tensor<5xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<5xf32>, tensor<?xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

func.func @test_add_1d_broadcast_static_to_dynamic(%arg0: tensor<1xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<1xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_add_1d_broadcast_static_to_static(%arg0: tensor<1xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<1xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @test_add_1d_matching_static(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @test_add_2d_all_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_add_2d_different_ranks(%arg0: tensor<3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

func.func @test_select_2d_one_dynamic(%arg0: tensor<2x?xi1>, %arg1: tensor<2x?xf32>, %arg2: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<2x?xi1>, tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}

// -----

func.func @test_simple_f32(%arg0: tensor<1xf32>) -> () {
  %0 = tosa.tanh %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %1 = tosa.abs %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %2 = tosa.add %0, %0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = tosa.sub %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %4 = tosa.mul %0, %1 {shift = 0 : i8} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %5 = tosa.negate %0 : (tensor<1xf32>) -> tensor<1xf32>
  %6 = tosa.pow %1, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %7 = tosa.rsqrt %1 : (tensor<1xf32>) -> tensor<1xf32>
  %8 = tosa.log %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %9 = tosa.exp %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %10 = tosa.greater %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %11 = tosa.greater_equal %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %12 = tosa.equal %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %13 = tosa.select %10, %0, %1 : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %14 = tosa.maximum %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %15 = tosa.minimum %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %16 = tosa.ceil %0 : (tensor<1xf32>) -> tensor<1xf32>
  %17 = tosa.floor %0 : (tensor<1xf32>) -> tensor<1xf32>
  %18 = tosa.clamp %0 {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xf32>) -> tensor<1xf32>
  %19 = tosa.sigmoid %0 : (tensor<1xf32>) -> tensor<1xf32>
  %20 = tosa.cast %0 : (tensor<1xf32>) -> tensor<1xi32>
  %21 = tosa.cast %0 : (tensor<1xf32>) -> tensor<1xi1>
  %22 = tosa.cast %0 : (tensor<1xf32>) -> tensor<1xf16>
  %23 = tosa.reciprocal %0 : (tensor<1xf32>) -> tensor<1xf32>
  %24 = tosa.erf %0 : (tensor<1xf32>) -> tensor<1xf32>
  return
}

// -----

func.func @test_simple_f16(%arg0: tensor<1xf16>) -> () {
  %0 = tosa.cast %arg0 : (tensor<1xf16>) -> tensor<1xf32>
  %1 = "tosa.cast"(%arg0) : (tensor<1xf16>) -> tensor<1xi8>
  %2 = "tosa.cast"(%arg0) : (tensor<1xf16>) -> tensor<1xi32>
  return
}

// -----

func.func @test_simple_i16(%arg0: tensor<1xi16>) -> () {
  %0 = tosa.mul %arg0, %arg0 {shift = 0 : i8} : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi32>
  return
}

// -----

func.func @test_simple_ui8(%arg0: tensor<1xui8>) -> () {
  %0 = tosa.cast %arg0 : (tensor<1xui8>) -> tensor<1xf32>
  return
}

// -----

func.func @test_simple_i32(%arg0: tensor<1xi32>) -> () {
  %0 = tosa.add %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = tosa.sub %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2 = tosa.mul %arg0, %arg0 {shift = 0 : i8} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %3 = tosa.mul %arg0, %arg0 {shift = 2 : i8} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %40 = tosa.int_div %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %5 = tosa.negate %arg0 : (tensor<1xi32>) -> tensor<1xi32>
  %6 = tosa.bitwise_and %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %7 = tosa.bitwise_or %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %8 = tosa.bitwise_xor %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %9 = tosa.logical_left_shift %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %10 = tosa.logical_right_shift %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %11 = tosa.arithmetic_right_shift %arg0, %arg0 {round = 0 : i1} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %12 = tosa.arithmetic_right_shift %arg0, %arg0 {round = 1 : i1} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %13 = tosa.clz %arg0 : (tensor<1xi32>) -> tensor<1xi32>
  %14 = tosa.greater %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %15 = tosa.greater_equal %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %16 = tosa.select %14, %0, %1 : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %17 = tosa.maximum %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %18 = tosa.minimum %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %19 = tosa.clamp %0 {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xi32>) -> tensor<1xi32>
  %20 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xi16>
  %21 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xi64>
  %22 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xi1>
  %23 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xf32>
  %24 = tosa.abs %arg0 : (tensor<1xi32>) -> tensor<1xi32>
  return
}

// -----

func.func @test_simple_ui8(%arg0: tensor<1xi8>) -> () {
  %0 = tosa.cast %arg0 : (tensor<1xi8>) -> tensor<1xf32>
  return
}

// -----

func.func @test_i8(%arg0: tensor<1xi8>) -> () {
  %0 = tosa.clamp %arg0 {min_int = -127 : i64, max_int = 126 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi8>) -> tensor<1xi8>
  %1 = tosa.clamp %arg0 {min_int = -130 : i64, max_int = 130 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi8>) -> tensor<1xi8>
  return
}

// -----

func.func @test_i64(%arg0: tensor<1xi64>) -> () {
  %0 = tosa.clamp %arg0 {min_int = -9223372036854775808 : i64, max_int = 9223372036854775807 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi64>) -> tensor<1xi64>
  return
}

// -----

func.func @test_clamp_f16(%arg0: tensor<1xf16>) -> () {
  %0 = tosa.clamp %arg0 {min_int = 0 : i64, max_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 6.0 : f32} : (tensor<1xf16>) -> tensor<1xf16>
  return
}

// -----

func.func @test_bool(%arg0: tensor<1xi1>, %arg1: tensor<1xi1>) -> () {
  %0 = tosa.logical_and %arg0, %arg1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
  %1 = tosa.logical_or %arg0, %arg1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
  %2 = tosa.logical_xor %arg0, %arg1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
  %3 = tosa.logical_not %arg0 : (tensor<1xi1>) -> tensor<1xi1>
  return
}

// -----

func.func @test_negate_quantized(%arg0: tensor<1xi8>) -> () {
  %0 = tosa.negate %arg0 {quantization_info = #tosa.unary_quant<input_zp = 0, output_zp = 0>} : (tensor<1xi8>) -> tensor<1xi8>
  %1 = tosa.negate %arg0 {quantization_info = #tosa.unary_quant<input_zp = 32639, output_zp = 0>} : (tensor<1xi8>) -> tensor<1xi8>
  %2 = tosa.negate %arg0 {quantization_info = #tosa.unary_quant<input_zp = 32640, output_zp = 0>} : (tensor<1xi8>) -> tensor<1xi8>
  return
}

// -----

func.func @test_identity(%arg0: tensor<1xf32>, %arg1: tensor<1xi32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0 = tosa.identity %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %1 = tosa.identity %arg1 : (tensor<1xi32>) -> tensor<1xi32>
  return %0, %1 : tensor<1xf32>, tensor<1xi32>
}

// -----

func.func @reduce_float(%arg0: tensor<5x4xf32>) -> () {
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  %1 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<5x4xf32>) -> tensor<5x1xf32>
  %2 = tosa.reduce_prod %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  %3 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  %4 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

func.func @reduce_float_dyn(%arg0: tensor<?x5x4xf32>) -> () {
  %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<?x5x4xf32>) -> tensor<?x1x4xf32>
  return
}

// -----

func.func @reduce_float_dyn_rank_1(%arg0: tensor<?xf32>) -> () {
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<?xf32>) -> tensor<1xf32>
  return
}

// -----

func.func @reduce_float_dyn_nonzero_batch(%arg0: tensor<5x?x4xf32>) -> () {
  %0 = tosa.reduce_prod %arg0 {axis = 2 : i32} : (tensor<5x?x4xf32>) -> tensor<5x?x1xf32>
  return
}

// -----

func.func @reduce_float_dyn_multiple(%arg0: tensor<?x?xf32>) -> () {
  %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<?x?xf32>) -> tensor<?x1xf32>
  return
}

// -----

func.func @reduce_int(%arg0: tensor<5x4xi32>) -> () {
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>
  %1 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<5x4xi32>) -> tensor<5x1xi32>
  %2 = tosa.reduce_prod %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>
  %3 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>
  %4 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>
  return
}

// -----

func.func @reduce_bool(%arg0: tensor<5x4xi1>) -> () {
  %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<5x4xi1>) -> tensor<1x4xi1>
  %1 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<5x4xi1>) -> tensor<1x4xi1>
  return
}

// -----

func.func @rescale_i8(%arg0 : tensor<2xi8>) -> () {
  %0 = tosa.rescale %arg0 {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = array<i32: 19689>, shift = array<i8: 15>, scale32 = false, double_round = false, per_channel = false} : (tensor<2xi8>) -> tensor<2xi8>
  %1 = tosa.rescale %arg0 {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = array<i32: 19689>, shift = array<i8: 15>, scale32 = false, double_round = false, per_channel = false} : (tensor<2xi8>) -> tensor<2xui8>
  return
}

// -----

func.func @rescale_i8_dyn_batch(%arg0 : tensor<?x2xi8>) -> () {
  %0 = tosa.rescale %arg0 {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = array<i32: 19689>, shift = array<i8: 15>, scale32 = false, double_round = false, per_channel = false} : (tensor<?x2xi8>) -> tensor<?x2xi8>
  %1 = tosa.rescale %arg0 {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = array<i32: 19689>, shift = array<i8: 15>, scale32 = false, double_round = false, per_channel = false} : (tensor<?x2xi8>) -> tensor<?x2xui8>
  return
}

// -----

func.func @rescale_dyn(%arg0 : tensor<1x?x?x32xi32>) -> () {
  %0 = tosa.rescale %arg0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1376784203>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 38>} : (tensor<1x?x?x32xi32>) -> tensor<1x?x?x32xi8>
  return
}

// -----

func.func @rescale_ui8(%arg0 : tensor<2xui8>) -> () {
  %0 = tosa.rescale %arg0 {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = array<i32: 19689>, shift = array<i8: 15>, scale32 = false, double_round = false, per_channel = false} : (tensor<2xui8>) -> tensor<2xi8>
  return
}

// -----

func.func @rescale_per_channel(%arg0 : tensor<3xi8>) -> (tensor<3xi8>) {
  %0 = tosa.rescale %arg0 {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = array<i32: 42, 43, 44>, shift = array<i8: 14, 15, 64>, scale32 = false, double_round = false, per_channel = false} : (tensor<3xi8>) -> tensor<3xi8>
  return %0 : tensor<3xi8>
}

// -----

func.func @rescaleDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  %0 = tosa.rescale %arg0 {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = array<i32: 19689>, shift = array<i8: 33>, scale32 = true, double_round = true, per_channel = false} : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

func.func @rescaleUnnecessaryDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  %0 = tosa.rescale %arg0 {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = array<i32: 19689>, shift = array<i8: 15>, scale32 = true, double_round = true, per_channel = false} : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

func.func @reverse(%arg0: tensor<5x4xi32>) -> () {
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<5x4xi32>
  %1 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<5x4xi32>) -> tensor<5x4xi32>
  return
}

// -----

func.func @reverse_dyn(%arg0: tensor<?xi32>) -> () {
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<?xi32>
  return
}

// -----

func.func @tile(%arg0 : tensor<2x3xi8>) -> () {
  %0 = tosa.tile %arg0 {multiples = array<i64: 2, 1>} : (tensor<2x3xi8>) -> tensor<4x3xi8>
  %1 = tosa.tile %arg0 {multiples = array<i64: 1, 2>} : (tensor<2x3xi8>) -> tensor<2x6xi8>
  %2 = tosa.tile %arg0 {multiples = array<i64: 5, 7>} : (tensor<2x3xi8>)  -> tensor<10x21xi8>
  return
}

// -----

func.func @tile_dyn_input(%arg0 : tensor<?x3xi8>) -> () {
  %0 = tosa.tile %arg0 {multiples = array<i64: 2, 1>} : (tensor<?x3xi8>)  -> tensor<?x3xi8>
  return
}

// -----

func.func @tile_dyn_multiples(%arg0 : tensor<2x3xi8>) -> () {
  %0 = tosa.tile %arg0 {multiples = array<i64: 2, -1>} : (tensor<2x3xi8>)  -> tensor<2x?xi8>
  return
}

// -----

func.func @argmax(%arg0 : tensor<3x2xi32>, %arg1 : tensor<6xf32>) -> () {
  %0 = tosa.argmax %arg0 { axis = 0 : i32} : (tensor<3x2xi32>)  -> tensor<2xi32>
  %1 = tosa.argmax %arg0 { axis = 1 : i32} : (tensor<3x2xi32>)  -> tensor<3xi32>
  %2 = tosa.argmax %arg1 { axis = 0 : i32} : (tensor<6xf32>)  -> tensor<i32>
  return
}

// -----

func.func @argmax_dyn_non_axis(%arg0 : tensor<3x?xi32>) -> () {
  %0 = tosa.argmax %arg0 { axis = 0 : i32} : (tensor<3x?xi32>)  -> tensor<?xi32>
  return
}

// -----

func.func @argmax_dyn_axis(%arg0 : tensor<3x?xi32>) -> () {
  %0 = tosa.argmax %arg0 { axis = 1 : i32} : (tensor<3x?xi32>)  -> tensor<3xi32>
  return
}

// -----

func.func @gather_float(%arg0: tensor<2x3x2xf32>, %arg1: tensor<2x3xi32>) -> () {
  %0 = tosa.gather %arg0, %arg1 : (tensor<2x3x2xf32>, tensor<2x3xi32>)  -> tensor<2x3x2xf32>
  return
}

// -----

func.func @gather_float_dyn(%arg0: tensor<?x3x2xf32>, %arg1: tensor<?x3xi32>) -> () {
  %0 = tosa.gather %arg0, %arg1 : (tensor<?x3x2xf32>, tensor<?x3xi32>)  -> tensor<?x3x2xf32>
  return
}

// -----

func.func @gather_float_all_dynamic(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xi32>) -> () {
  %0 = tosa.gather %arg0, %arg1 : (tensor<?x?x?xf32>, tensor<?x?xi32>)  -> tensor<?x?x?xf32>
  return
}

// -----

func.func @gather_int(%arg0: tensor<2x3x2xi32>, %arg1: tensor<2x3xi32>) -> () {
  %0 = tosa.gather %arg0, %arg1 : (tensor<2x3x2xi32>, tensor<2x3xi32>)  -> tensor<2x3x2xi32>
  return
}

// -----

func.func @table8(%arg0: tensor<6xi8>, %arg1: tensor<512xi8>) -> () {
  %0 = tosa.table %arg0, %arg1 : (tensor<6xi8>, tensor<512xi8>)  -> tensor<6xi8>
  return
}

// -----

func.func @table16(%arg0: tensor<6xi16>, %arg1: tensor<513xi16>) -> () {
  %0 = tosa.table %arg0, %arg1 : (tensor<6xi16>, tensor<513xi16>)  -> tensor<6xi32>
  return
}

// -----

func.func @table8_dyn(%arg0: tensor<?xi8>, %arg1: tensor<512xi8>) -> () {
  %0 = tosa.table %arg0, %arg1 : (tensor<?xi8>, tensor<512xi8>)  -> tensor<?xi8>
  return
}

// -----

func.func @table8_dyn_table(%arg0: tensor<6xi8>, %arg1: tensor<?xi8>) -> () {
  %0 = tosa.table %arg0, %arg1 : (tensor<6xi8>, tensor<?xi8>)  -> tensor<6xi8>
  return
}

// -----

func.func @test_static_rfft2d(%arg0: tensor<5x5x8xf32>) -> (tensor<5x5x5xf32>, tensor<5x5x5xf32>) {
  %output_real, %output_imag = "tosa.rfft2d"(%arg0) {} : (tensor<5x5x8xf32>) -> (tensor<5x5x5xf32>, tensor<5x5x5xf32>)
  return %output_real, %output_imag : tensor<5x5x5xf32>, tensor<5x5x5xf32>
}

// -----

func.func @test_dynamic_rfft2d(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %output_real, %output_imag = "tosa.rfft2d"(%arg0) {} : (tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return %output_real, %output_imag : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}

// -----

func.func @test_static_fft2d(%arg0: tensor<8x8x8xf32>, %arg1: tensor<8x8x8xf32>) -> (tensor<8x8x8xf32>, tensor<8x8x8xf32>) {
  %output_real, %output_imag = "tosa.fft2d"(%arg0, %arg1) {inverse=false} : (tensor<8x8x8xf32>, tensor<8x8x8xf32>) -> (tensor<8x8x8xf32>, tensor<8x8x8xf32>)
  return %output_real, %output_imag : tensor<8x8x8xf32>, tensor<8x8x8xf32>
}

// -----

func.func @test_dynamic_fft2d(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %output_real, %output_imag = "tosa.fft2d"(%arg0, %arg1) {inverse = true} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return %output_real, %output_imag : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}


