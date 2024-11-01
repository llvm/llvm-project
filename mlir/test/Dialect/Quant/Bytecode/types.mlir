// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// AnyQuantized
//===----------------------------------------------------------------------===//

// CHECK-LABEL: parseAnyFullySpecified
module @parseAnyFullySpecified attributes {
  // CHECK: bytecode.test = !quant.any<i8<-8:7>:f32>
  bytecode.test = !quant.any<i8<-8:7>:f32>
} {}

// CHECK-LABEL: parseAnyNoExpressedType
module @parseAnyNoExpressedType attributes {
  // CHECK: bytecode.test = !quant.any<i8<-8:7>>
  bytecode.test = !quant.any<i8<-8:7>>
} {}

// CHECK-LABEL: parseAnyOnlyStorageType
module @parseAnyOnlyStorageType attributes {
  // CHECK: bytecode.test = !quant.any<i8<-8:7>>
  bytecode.test = !quant.any<i8<-8:7>>
} {}

//===----------------------------------------------------------------------===//
// CalibratedQuantized
//===----------------------------------------------------------------------===//

// CHECK-LABEL: parseCalibrated
module @parseCalibrated attributes {
  // CHECK: !quant.calibrated<f32<-0.998:1.232100e+00>>
  bytecode.test = !quant.calibrated<f32<-0.998:1.2321>>
} {}

//===----------------------------------------------------------------------===//
// UniformQuantized
//===----------------------------------------------------------------------===//

// CHECK-LABEL: parseUniformPerLayer
module @parseUniformPerLayer attributes {
  // CHECK: !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
  bytecode.test = !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
} {}

//===----------------------------------------------------------------------===//
// UniformQuantizedPerAxis
//===----------------------------------------------------------------------===//

// CHECK-LABEL: parseUniformPerAxisScaleZero
module @parseUniformPerAxisScaleZero attributes {
  // CHECK: !quant.uniform<u8:f32:1, {2.000000e+02:-120,9.987200e-01:127}>
  bytecode.test = !quant.uniform<u8:f32:1, {2.000000e+02:-120,9.987200e-01:127}>
} {}

// CHECK-LABEL: parseUniformPerAxisScaleNoZero
module @parseUniformPerAxisScaleNoZero attributes {
  // CHECK: !quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01}>
  bytecode.test = !quant.uniform<i8:f32:1, {2.0e+2,0.99872}>
} {}

// CHECK-LABEL: parseUniformPerAxisMixed
module @parseUniformPerAxisMixed attributes {
  // CHECK: !quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01:120}>
  bytecode.test = !quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>
} {}

