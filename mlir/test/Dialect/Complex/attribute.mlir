// RUN: mlir-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

func.func @number_attr_f64() {
  "test.number_attr"() {
    // CHECK: attr = #complex.number<:f64 1.000000e+00, 0.000000e+00> : complex<f64>
    attr = #complex.number<:f64 1.0, 0.0>
  } : () -> ()

  return
}

func.func @number_attr_f32() {
  "test.number_attr"() {
    // CHECK: attr = #complex.number<:f32 1.000000e+00, 0.000000e+00> : complex<f32>
    attr = #complex.number<:f32 1.0, 0.0>
  } : () -> ()

  return
}
