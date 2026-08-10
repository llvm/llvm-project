// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold %s | FileCheck %s

// Skip big-endian platforms for dense resources
// XFAIL: target={{(s390x|sparc.*)-.*}}
// XFAIL: system-aix

// CHECK-LABEL: @transpose_fold_dense_resource
func.func @transpose_fold_dense_resource() -> tensor<2x2xf32> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>

  // CHECK-NOT: tosa.transpose
  %2 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x040000003f800000400000004040000040800000"
    }
  }
#-}

// -----

// CHECK-LABEL: @transpose_fold_dense_resource_f8e4m3fn
func.func @transpose_fold_dense_resource_f8e4m3fn() -> tensor<2x2xf8E4M3FN> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xf8E4M3FN>}> : () -> tensor<2x2xf8E4M3FN>

  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]> : tensor<2x2xf8E4M3FN>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xf8E4M3FN>) -> tensor<2x2xf8E4M3FN>
  // CHECK: return %[[CST]]
  return %1 : tensor<2x2xf8E4M3FN>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x0100000038404448"
    }
  }
#-}

// -----

// CHECK-LABEL: @transpose_fold_dense_resource_f8e5m2
func.func @transpose_fold_dense_resource_f8e5m2() -> tensor<2x2xf8E5M2> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xf8E5M2>}> : () -> tensor<2x2xf8E5M2>

  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]> : tensor<2x2xf8E5M2>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xf8E5M2>) -> tensor<2x2xf8E5M2>
  // CHECK: return %[[CST]]
  return %1 : tensor<2x2xf8E5M2>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x010000003c404244"
    }
  }
#-}

// -----

// CHECK-LABEL: @transpose_fold_dense_resource_f4e2m1fn
func.func @transpose_fold_dense_resource_f4e2m1fn() -> tensor<2x2xf4E2M1FN> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xf4E2M1FN>}> : () -> tensor<2x2xf4E2M1FN>

  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]> : tensor<2x2xf4E2M1FN>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xf4E2M1FN>) -> tensor<2x2xf4E2M1FN>
  // CHECK: return %[[CST]]
  return %1 : tensor<2x2xf4E2M1FN>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x0100000002040506"
    }
  }
#-}

// -----

// CHECK-LABEL: @transpose_fold_dense_resource_f16
func.func @transpose_fold_dense_resource_f16() -> tensor<2x2xf16> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xf16>}> : () -> tensor<2x2xf16>

  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]> : tensor<2x2xf16>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xf16>) -> tensor<2x2xf16>
  // CHECK: return %[[CST]]
  return %1 : tensor<2x2xf16>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x02000000003c004000420044"
    }
  }
#-}

// -----

// CHECK-LABEL: @transpose_fold_dense_resource_bf16
func.func @transpose_fold_dense_resource_bf16() -> tensor<2x2xbf16> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xbf16>}> : () -> tensor<2x2xbf16>

  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]> : tensor<2x2xbf16>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xbf16>) -> tensor<2x2xbf16>
  // CHECK: return %[[CST]]
  return %1 : tensor<2x2xbf16>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x02000000803f004040408040"
    }
  }
#-}

// -----

// CHECK-LABEL: @transpose_fold_dense_resource_f64
func.func @transpose_fold_dense_resource_f64() -> tensor<2x2xf64> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>

  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xf64>) -> tensor<2x2xf64>
  // CHECK: return %[[CST]]
  return %1 : tensor<2x2xf64>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x08000000000000000000f03f000000000000004000000000000008400000000000001040"
    }
  }
#-}
