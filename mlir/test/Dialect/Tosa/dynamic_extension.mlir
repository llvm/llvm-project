//--------------------------------------------------------
// Check operations when the dynamic extension is enabled.
//--------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="profiles=pro_int,pro_fp extensions=dynamic" -tosa-validate="strict-op-spec-alignment allow-invalid-op-datatype-combinations"

// -----

func.func @test_mul_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<13x1x3xi8>, %shift: tensor<1xi8>) -> tensor<13x21x3xi16> {
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi8>, tensor<13x1x3xi8>, tensor<1xi8>) -> tensor<13x21x3xi16>
  return %0 : tensor<13x21x3xi16>
}

// -----

func.func @test_table_non_const(%arg0 : tensor<4x5xi8>, %arg1 : tensor<256xi8>) -> () {
  %0 = tosa.table %arg0, %arg1 : (tensor<4x5xi8>, tensor<256xi8>) -> tensor<4x5xi8>
  return
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<1xi8>) -> tensor<13x22x4xi8> {
  %0 = tosa.const_shape {values = dense<[0, 0, 0, 1, 0, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  %1 = tosa.pad %arg0, %0, %arg1 : (tensor<13x21x3xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<13x22x4xi8>
  return %1 : tensor<13x22x4xi8>
}

// -----

func.func @test_rescale_non_const_multiplier(%arg0: tensor<13x21x3xi32>, %multiplier: tensor<1xi32>) -> tensor<13x21x3xi32> {
  %zps = "tosa.const"() {values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %zps, %zps {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----

func.func @test_rescale_non_const_shift(%arg0: tensor<13x21x3xi32>, %shift: tensor<1xi8>) -> tensor<13x21x3xi32> {
  %zps = "tosa.const"() {values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %zps, %zps {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----

func.func @test_rescale_non_const_input_zp(%arg0: tensor<13x21x3xi32>, %input_zp: tensor<1xi32>) -> tensor<13x21x3xi32> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %output_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = true, rounding_mode = SINGLE_ROUND, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----

func.func @test_rescale_non_const_output_zp(%arg0: tensor<13x21x3xi32>, %output_zp: tensor<1xi32>) -> tensor<13x21x3xi32> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = true, rounding_mode = SINGLE_ROUND, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----

func.func @test_matmul_non_const_zps(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>, %a_zp: tensor<1xf32>, %b_zp: tensor<1xf32>) -> tensor<1x14x28xf32> {
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----

func.func @test_negate_non_const_zps(%arg0: tensor<1xf32>, %input1_zp: tensor<1xf32>, %output_zp: tensor<1xf32>) -> tensor<1xf32> {
  %0 = tosa.negate %arg0, %input1_zp, %output_zp {} : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

func.func @test_avg_pool2d_non_const_zps(%arg0: tensor<1x32x32x8xf32>, %input_zp: tensor<1xf32>, %output_zp: tensor<1xf32>) -> tensor<1x32x32x8xf32> {
  %0 = "tosa.avg_pool2d"(%arg0, %input_zp, %output_zp) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}
