//--------------------------------------------------------------------------------------------------
// Test valid IR in terms of the shape and type of tensor, and the argument type of
// operation. Excludes the profile compilance checking since it is performed earlier in the
// validation flow.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics --tosa-validate="profile=pro_int,pro_fp extension=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,doubleround,inexactround" | FileCheck %s

// -----

// CHECK-LABEL: test_rescale_input_unsigned
func.func @test_rescale_input_unsigned(%arg0: tensor<1x1xui8>) -> (tensor<1x1xi8>) {
  %0 = "tosa.const"() <{values = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = "tosa.const"() <{values = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
  %2 = "tosa.const"() <{values = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
  %r = tosa.rescale %arg0, %1, %0, %3, %2 {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true} : (tensor<1x1xui8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1xi8>
  return %r : tensor<1x1xi8>
}

// -----

// CHECK-LABEL: test_rescale_output_unsigned
func.func @test_rescale_output_unsigned(%arg0: tensor<1x1xi8>) -> (tensor<1x1xui8>) {
  %0 = "tosa.const"() <{values = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = "tosa.const"() <{values = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
  %2 = "tosa.const"() <{values = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
  %r = tosa.rescale %arg0, %1, %0, %3, %2 {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true} : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1xui8>
  return %r : tensor<1x1xui8>
}
