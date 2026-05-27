// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.TOSA.Erf
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @erf_fp
func.func @erf_fp(%arg0: tensor<47x38x51xf32>) -> tensor<47x38x51xf32> {
  // CHECK: %[[ERF:.*]] = spirv.Tosa.Erf  %arg0 : !spirv.arm.tensor<47x38x51xf32> -> !spirv.arm.tensor<47x38x51xf32>
  %res = tosa.erf %arg0  : (tensor<47x38x51xf32>) -> tensor<47x38x51xf32>
  return %res : tensor<47x38x51xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sigmoid
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @sigmoid_fp
func.func @sigmoid_fp(%arg0: tensor<28x43x45xf32>) -> tensor<28x43x45xf32> {
  // CHECK: %[[SIGMOID:.*]] = spirv.Tosa.Sigmoid  %arg0 : !spirv.arm.tensor<28x43x45xf32> -> !spirv.arm.tensor<28x43x45xf32>
  %res = tosa.sigmoid %arg0  : (tensor<28x43x45xf32>) -> tensor<28x43x45xf32>
  return %res : tensor<28x43x45xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tanh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @tanh_fp
func.func @tanh_fp(%arg0: tensor<46x50x36xf16>) -> tensor<46x50x36xf16> {
  // CHECK: %[[TANH:.*]] = spirv.Tosa.Tanh  %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf16>
  %res = tosa.tanh %arg0  : (tensor<46x50x36xf16>) -> tensor<46x50x36xf16>
  return %res : tensor<46x50x36xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @add_int
func.func @add_int(%arg0: tensor<4x7x3x10xi32>, %arg1: tensor<4x7x3x1xi32>) -> tensor<4x7x3x10xi32> {
  // CHECK: %[[ADD:.*]] = spirv.Tosa.Add  %arg0, %arg1 : !spirv.arm.tensor<4x7x3x10xi32>, !spirv.arm.tensor<4x7x3x1xi32> -> !spirv.arm.tensor<4x7x3x10xi32>
  %res = tosa.add %arg0, %arg1  : (tensor<4x7x3x10xi32>, tensor<4x7x3x1xi32>) -> tensor<4x7x3x10xi32>
  return %res : tensor<4x7x3x10xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwiseand_int
func.func @bitwiseand_int(%arg0: tensor<4x1x7x12xi16>, %arg1: tensor<4x13x7x12xi16>) -> tensor<4x13x7x12xi16> {
  // CHECK: %[[BITWISEAND:.*]] = spirv.Tosa.BitwiseAnd  %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
  %res = tosa.bitwise_and %arg0, %arg1  : (tensor<4x1x7x12xi16>, tensor<4x13x7x12xi16>) -> tensor<4x13x7x12xi16>
  return %res : tensor<4x13x7x12xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwiseor_int
func.func @bitwiseor_int(%arg0: tensor<11x30x23xi32>, %arg1: tensor<1x30x23xi32>) -> tensor<11x30x23xi32> {
  // CHECK: %[[BITWISEOR:.*]] = spirv.Tosa.BitwiseOr  %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
  %res = tosa.bitwise_or %arg0, %arg1  : (tensor<11x30x23xi32>, tensor<1x30x23xi32>) -> tensor<11x30x23xi32>
  return %res : tensor<11x30x23xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwisexor_int
func.func @bitwisexor_int(%arg0: tensor<4x8x13x9xi16>, %arg1: tensor<4x8x1x9xi16>) -> tensor<4x8x13x9xi16> {
  // CHECK: %[[BITWISEXOR:.*]] = spirv.Tosa.BitwiseXor  %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
  %res = tosa.bitwise_xor %arg0, %arg1  : (tensor<4x8x13x9xi16>, tensor<4x8x1x9xi16>) -> tensor<4x8x13x9xi16>
  return %res : tensor<4x8x13x9xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.IntDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @intdiv_any
func.func @intdiv_any(%arg0: tensor<1x65533x1xi32>, %arg1: tensor<2x65533x1xi32>) -> tensor<2x65533x1xi32> {
  // CHECK: %[[INTDIV:.*]] = spirv.Tosa.IntDiv  %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
  %res = tosa.intdiv %arg0, %arg1  : (tensor<1x65533x1xi32>, tensor<2x65533x1xi32>) -> tensor<2x65533x1xi32>
  return %res : tensor<2x65533x1xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicaland_any
func.func @logicaland_any(%arg0: tensor<2x1x7x11xi1>, %arg1: tensor<2x4x7x11xi1>) -> tensor<2x4x7x11xi1> {
  // CHECK: %[[LOGICALAND:.*]] = spirv.Tosa.LogicalAnd  %arg0, %arg1 : !spirv.arm.tensor<2x1x7x11xi1>, !spirv.arm.tensor<2x4x7x11xi1> -> !spirv.arm.tensor<2x4x7x11xi1>
  %res = tosa.logical_and %arg0, %arg1  : (tensor<2x1x7x11xi1>, tensor<2x4x7x11xi1>) -> tensor<2x4x7x11xi1>
  return %res : tensor<2x4x7x11xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalLeftShift
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalleftshift_any
func.func @logicalleftshift_any(%arg0: tensor<7x1x11x4xi8>, %arg1: tensor<7x8x11x4xi8>) -> tensor<7x8x11x4xi8> {
  // CHECK: %[[LOGICALLEFTSHIFT:.*]] = spirv.Tosa.LogicalLeftShift  %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
  %res = tosa.logical_left_shift %arg0, %arg1  : (tensor<7x1x11x4xi8>, tensor<7x8x11x4xi8>) -> tensor<7x8x11x4xi8>
  return %res : tensor<7x8x11x4xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalRightShift
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalrightshift_any
func.func @logicalrightshift_any(%arg0: tensor<6x13x1x19xi8>, %arg1: tensor<6x13x6x19xi8>) -> tensor<6x13x6x19xi8> {
  // CHECK: %[[LOGICALRIGHTSHIFT:.*]] = spirv.Tosa.LogicalRightShift  %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
  %res = tosa.logical_right_shift %arg0, %arg1  : (tensor<6x13x1x19xi8>, tensor<6x13x6x19xi8>) -> tensor<6x13x6x19xi8>
  return %res : tensor<6x13x6x19xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalor_any
func.func @logicalor_any(%arg0: tensor<3x6x12x5xi1>, %arg1: tensor<3x6x1x5xi1>) -> tensor<3x6x12x5xi1> {
  // CHECK: %[[LOGICALOR:.*]] = spirv.Tosa.LogicalOr  %arg0, %arg1 : !spirv.arm.tensor<3x6x12x5xi1>, !spirv.arm.tensor<3x6x1x5xi1> -> !spirv.arm.tensor<3x6x12x5xi1>
  %res = tosa.logical_or %arg0, %arg1  : (tensor<3x6x12x5xi1>, tensor<3x6x1x5xi1>) -> tensor<3x6x12x5xi1>
  return %res : tensor<3x6x12x5xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalxor_any
func.func @logicalxor_any(%arg0: tensor<11x4x9x12xi1>, %arg1: tensor<11x4x9x1xi1>) -> tensor<11x4x9x12xi1> {
  // CHECK: %[[LOGICALXOR:.*]] = spirv.Tosa.LogicalXor  %arg0, %arg1 : !spirv.arm.tensor<11x4x9x12xi1>, !spirv.arm.tensor<11x4x9x1xi1> -> !spirv.arm.tensor<11x4x9x12xi1>
  %res = tosa.logical_xor %arg0, %arg1  : (tensor<11x4x9x12xi1>, tensor<11x4x9x1xi1>) -> tensor<11x4x9x12xi1>
  return %res : tensor<11x4x9x12xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Maximum
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @maximum_int
func.func @maximum_int(%arg0: tensor<1x2x65533x1xi32>, %arg1: tensor<1x2x65533x2xi32>) -> tensor<1x2x65533x2xi32> {
  // CHECK: %[[MAXIMUM:.*]] = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x2x65533x1xi32>, !spirv.arm.tensor<1x2x65533x2xi32> -> !spirv.arm.tensor<1x2x65533x2xi32>
  %res = tosa.maximum %arg0, %arg1  {nan_mode = PROPAGATE} : (tensor<1x2x65533x1xi32>, tensor<1x2x65533x2xi32>) -> tensor<1x2x65533x2xi32>
  return %res : tensor<1x2x65533x2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Minimum
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @minimum_int
func.func @minimum_int(%arg0: tensor<15x2x10x11xi32>, %arg1: tensor<15x1x10x11xi32>) -> tensor<15x2x10x11xi32> {
  // CHECK: %[[MINIMUM:.*]] = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<15x2x10x11xi32>, !spirv.arm.tensor<15x1x10x11xi32> -> !spirv.arm.tensor<15x2x10x11xi32>
  %res = tosa.minimum %arg0, %arg1  {nan_mode = PROPAGATE} : (tensor<15x2x10x11xi32>, tensor<15x1x10x11xi32>) -> tensor<15x2x10x11xi32>
  return %res : tensor<15x2x10x11xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @pow_fp
func.func @pow_fp(%arg0: tensor<1x52x53xf16>, %arg1: tensor<44x52x53xf16>) -> tensor<44x52x53xf16> {
  // CHECK: %[[POW:.*]] = spirv.Tosa.Pow  %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
  %res = tosa.pow %arg0, %arg1  : (tensor<1x52x53xf16>, tensor<44x52x53xf16>) -> tensor<44x52x53xf16>
  return %res : tensor<44x52x53xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @sub_int
func.func @sub_int(%arg0: tensor<6x10x6x6xi32>, %arg1: tensor<1x10x6x6xi32>) -> tensor<6x10x6x6xi32> {
  // CHECK: %[[SUB:.*]] = spirv.Tosa.Sub  %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  %res = tosa.sub %arg0, %arg1  : (tensor<6x10x6x6xi32>, tensor<1x10x6x6xi32>) -> tensor<6x10x6x6xi32>
  return %res : tensor<6x10x6x6xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Abs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @abs_int
func.func @abs_int(%arg0: tensor<5x1x4x4xi32>) -> tensor<5x1x4x4xi32> {
  // CHECK: %[[ABS:.*]] = spirv.Tosa.Abs  %arg0 : !spirv.arm.tensor<5x1x4x4xi32> -> !spirv.arm.tensor<5x1x4x4xi32>
  %res = tosa.abs %arg0  : (tensor<5x1x4x4xi32>) -> tensor<5x1x4x4xi32>
  return %res : tensor<5x1x4x4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseNot
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwisenot_int
func.func @bitwisenot_int(%arg0: tensor<12x56x50xi32>) -> tensor<12x56x50xi32> {
  // CHECK: %[[BITWISENOT:.*]] = spirv.Tosa.BitwiseNot  %arg0 : !spirv.arm.tensor<12x56x50xi32> -> !spirv.arm.tensor<12x56x50xi32>
  %res = tosa.bitwise_not %arg0  : (tensor<12x56x50xi32>) -> tensor<12x56x50xi32>
  return %res : tensor<12x56x50xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Ceil
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @ceil_fp
func.func @ceil_fp(%arg0: tensor<46x55x53xf16>) -> tensor<46x55x53xf16> {
  // CHECK: %[[CEIL:.*]] = spirv.Tosa.Ceil  %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
  %res = tosa.ceil %arg0  : (tensor<46x55x53xf16>) -> tensor<46x55x53xf16>
  return %res : tensor<46x55x53xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clz
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @clz_int
func.func @clz_int(%arg0: tensor<14x10x7x5xi32>) -> tensor<14x10x7x5xi32> {
  // CHECK: %[[CLZ:.*]] = spirv.Tosa.Clz  %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
  %res = tosa.clz %arg0  : (tensor<14x10x7x5xi32>) -> tensor<14x10x7x5xi32>
  return %res : tensor<14x10x7x5xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @cos_fp
func.func @cos_fp(%arg0: tensor<44x49x51xf32>) -> tensor<44x49x51xf32> {
  // CHECK: %[[COS:.*]] = spirv.Tosa.Cos  %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
  %res = tosa.cos %arg0  : (tensor<44x49x51xf32>) -> tensor<44x49x51xf32>
  return %res : tensor<44x49x51xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Exp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @exp_fp
func.func @exp_fp(%arg0: tensor<37x53x47xf32>) -> tensor<37x53x47xf32> {
  // CHECK: %[[EXP:.*]] = spirv.Tosa.Exp  %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
  %res = tosa.exp %arg0  : (tensor<37x53x47xf32>) -> tensor<37x53x47xf32>
  return %res : tensor<37x53x47xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Floor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @floor_fp
func.func @floor_fp(%arg0: tensor<40x52x42xf32>) -> tensor<40x52x42xf32> {
  // CHECK: %[[FLOOR:.*]] = spirv.Tosa.Floor  %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
  %res = tosa.floor %arg0  : (tensor<40x52x42xf32>) -> tensor<40x52x42xf32>
  return %res : tensor<40x52x42xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Log
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @log_fp
func.func @log_fp(%arg0: tensor<45x43x36xf16>) -> tensor<45x43x36xf16> {
  // CHECK: %[[LOG:.*]] = spirv.Tosa.Log  %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
  %res = tosa.log %arg0  : (tensor<45x43x36xf16>) -> tensor<45x43x36xf16>
  return %res : tensor<45x43x36xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalNot
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalnot_any
func.func @logicalnot_any(%arg0: tensor<54x26x10xi1>) -> tensor<54x26x10xi1> {
  // CHECK: %[[LOGICALNOT:.*]] = spirv.Tosa.LogicalNot  %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x10xi1>
  %res = tosa.logical_not %arg0  : (tensor<54x26x10xi1>) -> tensor<54x26x10xi1>
  return %res : tensor<54x26x10xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reciprocal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reciprocal_fp
func.func @reciprocal_fp(%arg0: tensor<38x47x44xf32>) -> tensor<38x47x44xf32> {
  // CHECK: %[[RECIPROCAL:.*]] = spirv.Tosa.Reciprocal  %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
  %res = tosa.reciprocal %arg0  : (tensor<38x47x44xf32>) -> tensor<38x47x44xf32>
  return %res : tensor<38x47x44xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rsqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @rsqrt_fp
func.func @rsqrt_fp(%arg0: tensor<40x57x56xf32>) -> tensor<40x57x56xf32> {
  // CHECK: %[[RSQRT:.*]] = spirv.Tosa.Rsqrt  %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
  %res = tosa.rsqrt %arg0  : (tensor<40x57x56xf32>) -> tensor<40x57x56xf32>
  return %res : tensor<40x57x56xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @sin_fp
func.func @sin_fp(%arg0: tensor<49x38x58xf16>) -> tensor<49x38x58xf16> {
  // CHECK: %[[SIN:.*]] = spirv.Tosa.Sin  %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
  %res = tosa.sin %arg0  : (tensor<49x38x58xf16>) -> tensor<49x38x58xf16>
  return %res : tensor<49x38x58xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Equal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @equal_int
func.func @equal_int(%arg0: tensor<51x28x59xi32>, %arg1: tensor<51x1x59xi32>) -> tensor<51x28x59xi1> {
  // CHECK: %[[EQUAL:.*]] = spirv.Tosa.Equal  %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
  %res = tosa.equal %arg0, %arg1  : (tensor<51x28x59xi32>, tensor<51x1x59xi32>) -> tensor<51x28x59xi1>
  return %res : tensor<51x28x59xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Greater
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @greater_int
func.func @greater_int(%arg0: tensor<11x10x10x2xi32>, %arg1: tensor<11x10x10x1xi32>) -> tensor<11x10x10x2xi1> {
  // CHECK: %[[GREATER:.*]] = spirv.Tosa.Greater  %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
  %res = tosa.greater %arg0, %arg1  : (tensor<11x10x10x2xi32>, tensor<11x10x10x1xi32>) -> tensor<11x10x10x2xi1>
  return %res : tensor<11x10x10x2xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.GreaterEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @greaterequal_int
func.func @greaterequal_int(%arg0: tensor<10x17x7x1xi32>, %arg1: tensor<10x17x7x16xi32>) -> tensor<10x17x7x16xi1> {
  // CHECK: %[[GREATEREQUAL:.*]] = spirv.Tosa.GreaterEqual  %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
  %res = tosa.greater_equal %arg0, %arg1  : (tensor<10x17x7x1xi32>, tensor<10x17x7x16xi32>) -> tensor<10x17x7x16xi1>
  return %res : tensor<10x17x7x16xi1>
}

