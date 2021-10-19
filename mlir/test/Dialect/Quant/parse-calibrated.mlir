// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: parseCalibrated
// CHECK: !quant.calibrated<f32<-0.998:1.232100e+00>
!qalias = type !quant.calibrated<f32<-0.998:1.2321>>
func @parseCalibrated() -> !qalias {
  %0 = "foo"() : () -> !qalias
  return %0 : !qalias
}
