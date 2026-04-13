// RUN: mlir-opt %s -split-input-file -tosa-attach-target="profiles=pro_int,pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,doubleround,inexactround,dynamic level=none" | FileCheck %s --check-prefix=CHECK-ALL
// RUN: mlir-opt %s -split-input-file -tosa-attach-target="level=8k" | FileCheck %s --check-prefix=CHECK-LVL-8K
// RUN: mlir-opt %s -split-input-file -tosa-attach-target | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: mlir-opt %s -split-input-file -tosa-attach-target="specification_version=1.1.draft" | FileCheck %s --check-prefix=CHECK-VERSION-1P1

// -----

// CHECK-ALL: module attributes {tosa.target_env = #tosa.target_env<specification_version = "1.0", level = none, profiles = [pro_int, pro_fp], extensions = [int16, int4, bf16, fp8e4m3, fp8e5m2, fft, variable, controlflow, doubleround, inexactround, dynamic]>}
// CHECK-LVL-8K: module attributes {tosa.target_env = #tosa.target_env<specification_version = "1.0", level = "8k", profiles = [], extensions = []>}
// CHECK-DEFAULT: module attributes {tosa.target_env = #tosa.target_env<specification_version = "1.0", level = "8k", profiles = [], extensions = []>}
// CHECK-VERSION-1P1: module attributes {tosa.target_env = #tosa.target_env<specification_version = "1.1.draft", level = "8k", profiles = [], extensions = []>}
// CHECK-LABEL: test_simple
func.func @test_simple(%arg0 : tensor<1x1x1x1xf32>, %arg1 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
  %1 = tosa.add %arg0, %arg1 : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  return %1 : tensor<1x1x1x1xf32>
}
