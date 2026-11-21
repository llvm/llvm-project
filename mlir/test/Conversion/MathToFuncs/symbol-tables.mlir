// RUN: mlir-opt %s  -pass-pipeline="builtin.module(convert-math-to-funcs{convert-ctlz})" | FileCheck %s

// CHECK-LABEL: gpu.module @first {
// CHECK-LABEL: gpu.func @kernel(
// CHECK-COUNT-2: call @__mlir_math_ipowi_i32
// CHECK-COUNT-2: call @__mlir_math_ipowi_i64
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i64
// CHECK-COUNT-2: call @__mlir_math_fpowi_f64_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f64_i64
// CHECK-COUNT-2: call @__mlir_math_ctlz_i32
// CHECK-COUNT-2: call @__mlir_math_ctlz_i64
// CHECK-LABEL: gpu.func @kernel_vector(
// CHECK: vector.extract
// CHECK-COUNT-2: call @__mlir_math_ipowi_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i32
// CHECK-COUNT-8: func.func private @__mlir_math_
// CHECK-NOT: func.func private @__mlir_math_
// CHECK-LABEL: gpu.module @second {
// CHECK-LABEL: gpu.func @kernel(
// CHECK-COUNT-2: call @__mlir_math_ipowi_i32
// CHECK-COUNT-2: call @__mlir_math_ipowi_i64
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i64
// CHECK-COUNT-2: call @__mlir_math_fpowi_f64_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f64_i64
// CHECK-COUNT-2: call @__mlir_math_ctlz_i32
// CHECK-COUNT-2: call @__mlir_math_ctlz_i64
// CHECK-LABEL: gpu.func @kernel_vector(
// CHECK: vector.extract
// CHECK-COUNT-2: call @__mlir_math_ipowi_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i32
// CHECK-COUNT-8: func.func private @__mlir_math_
// CHECK-NOT: func.func private @__mlir_math_
// CHECK-LABEL: func.func @host(
// CHECK-COUNT-2: call @__mlir_math_ipowi_i32
// CHECK-COUNT-2: call @__mlir_math_ipowi_i64
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i64
// CHECK-COUNT-2: call @__mlir_math_fpowi_f64_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f64_i64
// CHECK-COUNT-2: call @__mlir_math_ctlz_i32
// CHECK-COUNT-2: call @__mlir_math_ctlz_i64
// CHECK-LABEL: func.func @host_vector(
// CHECK: vector.extract
// CHECK-COUNT-2: call @__mlir_math_ipowi_i32
// CHECK-COUNT-2: call @__mlir_math_fpowi_f32_i32
// CHECK-COUNT-8: func.func private @__mlir_math_
// CHECK-NOT: func.func private @__mlir_math_

module {
  gpu.module @first {
    gpu.func @kernel(%base32: i32, %exp32: i32, %base64: i64,
                     %exp64: i64, %basef32: f32, %basef64: f64) kernel {
      %ipowi32_0 = math.ipowi %base32, %exp32 : i32
      %ipowi32_1 = math.ipowi %base32, %exp32 : i32
      %ipowi64_0 = math.ipowi %base64, %exp64 : i64
      %ipowi64_1 = math.ipowi %base64, %exp64 : i64
      %fpowi_f32_i32_0 = math.fpowi %basef32, %exp32 : f32, i32
      %fpowi_f32_i32_1 = math.fpowi %basef32, %exp32 : f32, i32
      %fpowi_f32_i64_0 = math.fpowi %basef32, %exp64 : f32, i64
      %fpowi_f32_i64_1 = math.fpowi %basef32, %exp64 : f32, i64
      %fpowi_f64_i32_0 = math.fpowi %basef64, %exp32 : f64, i32
      %fpowi_f64_i32_1 = math.fpowi %basef64, %exp32 : f64, i32
      %fpowi_f64_i64_0 = math.fpowi %basef64, %exp64 : f64, i64
      %fpowi_f64_i64_1 = math.fpowi %basef64, %exp64 : f64, i64
      %ctlz32_0 = math.ctlz %base32 : i32
      %ctlz32_1 = math.ctlz %base32 : i32
      %ctlz64_0 = math.ctlz %base64 : i64
      %ctlz64_1 = math.ctlz %base64 : i64
      gpu.return
    }
    gpu.func @kernel_vector(%base32: vector<2xi32>, %exp32: vector<2xi32>,
                            %basef32: vector<2xf32>) kernel {
      %ipowi = math.ipowi %base32, %exp32 : vector<2xi32>
      %fpowi = math.fpowi %basef32, %exp32 : vector<2xf32>, vector<2xi32>
      gpu.return
    }
  }
  gpu.module @second {
    gpu.func @kernel(%base32: i32, %exp32: i32, %base64: i64,
                     %exp64: i64, %basef32: f32, %basef64: f64) kernel {
      %ipowi32_0 = math.ipowi %base32, %exp32 : i32
      %ipowi32_1 = math.ipowi %base32, %exp32 : i32
      %ipowi64_0 = math.ipowi %base64, %exp64 : i64
      %ipowi64_1 = math.ipowi %base64, %exp64 : i64
      %fpowi_f32_i32_0 = math.fpowi %basef32, %exp32 : f32, i32
      %fpowi_f32_i32_1 = math.fpowi %basef32, %exp32 : f32, i32
      %fpowi_f32_i64_0 = math.fpowi %basef32, %exp64 : f32, i64
      %fpowi_f32_i64_1 = math.fpowi %basef32, %exp64 : f32, i64
      %fpowi_f64_i32_0 = math.fpowi %basef64, %exp32 : f64, i32
      %fpowi_f64_i32_1 = math.fpowi %basef64, %exp32 : f64, i32
      %fpowi_f64_i64_0 = math.fpowi %basef64, %exp64 : f64, i64
      %fpowi_f64_i64_1 = math.fpowi %basef64, %exp64 : f64, i64
      %ctlz32_0 = math.ctlz %base32 : i32
      %ctlz32_1 = math.ctlz %base32 : i32
      %ctlz64_0 = math.ctlz %base64 : i64
      %ctlz64_1 = math.ctlz %base64 : i64
      gpu.return
    }
    gpu.func @kernel_vector(%base32: vector<2xi32>, %exp32: vector<2xi32>,
                            %basef32: vector<2xf32>) kernel {
      %ipowi = math.ipowi %base32, %exp32 : vector<2xi32>
      %fpowi = math.fpowi %basef32, %exp32 : vector<2xf32>, vector<2xi32>
      gpu.return
    }
  }
  func.func @host(%base32: i32, %exp32: i32, %base64: i64,
                  %exp64: i64, %basef32: f32, %basef64: f64) {
    %ipowi32_0 = math.ipowi %base32, %exp32 : i32
    %ipowi32_1 = math.ipowi %base32, %exp32 : i32
    %ipowi64_0 = math.ipowi %base64, %exp64 : i64
    %ipowi64_1 = math.ipowi %base64, %exp64 : i64
    %fpowi_f32_i32_0 = math.fpowi %basef32, %exp32 : f32, i32
    %fpowi_f32_i32_1 = math.fpowi %basef32, %exp32 : f32, i32
    %fpowi_f32_i64_0 = math.fpowi %basef32, %exp64 : f32, i64
    %fpowi_f32_i64_1 = math.fpowi %basef32, %exp64 : f32, i64
    %fpowi_f64_i32_0 = math.fpowi %basef64, %exp32 : f64, i32
    %fpowi_f64_i32_1 = math.fpowi %basef64, %exp32 : f64, i32
    %fpowi_f64_i64_0 = math.fpowi %basef64, %exp64 : f64, i64
    %fpowi_f64_i64_1 = math.fpowi %basef64, %exp64 : f64, i64
    %ctlz32_0 = math.ctlz %base32 : i32
    %ctlz32_1 = math.ctlz %base32 : i32
    %ctlz64_0 = math.ctlz %base64 : i64
    %ctlz64_1 = math.ctlz %base64 : i64
    func.return
  }
  func.func @host_vector(%base32: vector<2xi32>, %exp32: vector<2xi32>,
                         %basef32: vector<2xf32>) {
    %ipowi = math.ipowi %base32, %exp32 : vector<2xi32>
    %fpowi = math.fpowi %basef32, %exp32 : vector<2xf32>, vector<2xi32>
    func.return
  }
}
