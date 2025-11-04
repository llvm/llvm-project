// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// -----

// Test valid architectures work

// Valid case on sm_100a
gpu.module @valid_f16x2_rs_sm_100a [#nvvm.target<chip = "sm_100a">] {
  func.func @convert_rs() {
    %f1 = llvm.mlir.constant(1.0 : f32) : f32
    %f2 = llvm.mlir.constant(2.0 : f32) : f32
    %rbits = llvm.mlir.constant(0x12345678 : i32) : i32
    %res = nvvm.convert.f32x2.to.f16x2 %f1, %f2, %rbits : vector<2xf16>
    return
  }
}

// Valid case on sm_103a
gpu.module @valid_bf16x2_rs_sm_103a [#nvvm.target<chip = "sm_103a">] {
  func.func @convert_rs() {
    %f1 = llvm.mlir.constant(1.0 : f32) : f32
    %f2 = llvm.mlir.constant(2.0 : f32) : f32
    %rbits = llvm.mlir.constant(0 : i32) : i32
    %res = nvvm.convert.f32x2.to.bf16x2 %f1, %f2, %rbits : vector<2xbf16>
    return
  }
}

// -----

// Test F32x2 -> F16x2 with stochastic rounding (.rs)

// CHECK-LABEL: @convert_f32x2_to_f16x2_rs
llvm.func @convert_f32x2_to_f16x2_rs(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xf16> {
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB,  %rbits : vector<2xf16>
  llvm.return %res : vector<2xf16>
}

// CHECK-LABEL: @convert_f32x2_to_f16x2_rs_satfinite
llvm.func @convert_f32x2_to_f16x2_rs_satfinite(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xf16> {
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>
  llvm.return %res : vector<2xf16>
}

// CHECK-LABEL: @convert_f32x2_to_f16x2_rs_relu
llvm.func @convert_f32x2_to_f16x2_rs_relu(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xf16> {
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {relu = true} : vector<2xf16>
  llvm.return %res : vector<2xf16>
}

// CHECK-LABEL: @convert_f32x2_to_f16x2_rs_relu_satfinite
llvm.func @convert_f32x2_to_f16x2_rs_relu_satfinite(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xf16> {
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>
  llvm.return %res : vector<2xf16>
}

// -----

// Test F32x2 -> BF16x2 with stochastic rounding (.rs)

// CHECK-LABEL: @convert_f32x2_to_bf16x2_rs
llvm.func @convert_f32x2_to_bf16x2_rs(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xbf16> {
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB,  %rbits : vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// CHECK-LABEL: @convert_f32x2_to_bf16x2_rs_satfinite
llvm.func @convert_f32x2_to_bf16x2_rs_satfinite(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xbf16> {
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// CHECK-LABEL: @convert_f32x2_to_bf16x2_rs_relu
llvm.func @convert_f32x2_to_bf16x2_rs_relu(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xbf16> {
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {relu = true} : vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// CHECK-LABEL: @convert_f32x2_to_bf16x2_rs_relu_satfinite
llvm.func @convert_f32x2_to_bf16x2_rs_relu_satfinite(%srcA : f32, %srcB : f32, %rbits : i32) -> vector<2xbf16> {
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// -----

// Test F32x4 -> F8x4 (E4M3) with stochastic rounding (.rs)

// CHECK-LABEL: @convert_f32x4_to_f8x4_e4m3_rs
llvm.func @convert_f32x4_to_f8x4_e4m3_rs(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e4m3x4.rs.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f8x4 %src, %rbits : vector<4xf32> -> vector<4xi8> (f8E4M3FN)
  llvm.return %res : vector<4xi8>
}

// CHECK-LABEL: @convert_f32x4_to_f8x4_e4m3_rs_relu
llvm.func @convert_f32x4_to_f8x4_e4m3_rs_relu(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e4m3x4.rs.relu.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f8x4 %src, %rbits {relu = true} : vector<4xf32> -> vector<4xi8> (f8E4M3FN)
  llvm.return %res : vector<4xi8>
}

// -----

// Test F32x4 -> F8x4 (E5M2) with stochastic rounding (.rs)

// CHECK-LABEL: @convert_f32x4_to_f8x4_e5m2_rs
llvm.func @convert_f32x4_to_f8x4_e5m2_rs(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e5m2x4.rs.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f8x4 %src, %rbits : vector<4xf32> -> vector<4xi8> (f8E5M2)
  llvm.return %res : vector<4xi8>
}

// CHECK-LABEL: @convert_f32x4_to_f8x4_e5m2_rs_relu
llvm.func @convert_f32x4_to_f8x4_e5m2_rs_relu(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e5m2x4.rs.relu.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f8x4 %src, %rbits {relu = true} : vector<4xf32> -> vector<4xi8> (f8E5M2)
  llvm.return %res : vector<4xi8>
}

// -----

// Test F32x4 -> F6x4 (E2M3) with stochastic rounding (.rs)

// CHECK-LABEL: @convert_f32x4_to_f6x4_e2m3_rs
llvm.func @convert_f32x4_to_f6x4_e2m3_rs(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e2m3x4.rs.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f6x4 %src, %rbits : vector<4xf32> -> vector<4xi8> (f6E2M3FN)
  llvm.return %res : vector<4xi8>
}

// CHECK-LABEL: @convert_f32x4_to_f6x4_e2m3_rs_relu
llvm.func @convert_f32x4_to_f6x4_e2m3_rs_relu(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e2m3x4.rs.relu.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f6x4 %src, %rbits {relu = true} : vector<4xf32> -> vector<4xi8> (f6E2M3FN)
  llvm.return %res : vector<4xi8>
}

// -----

// Test F32x4 -> F6x4 (E3M2) with stochastic rounding (.rs)

// CHECK-LABEL: @convert_f32x4_to_f6x4_e3m2_rs
llvm.func @convert_f32x4_to_f6x4_e3m2_rs(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e3m2x4.rs.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f6x4 %src, %rbits : vector<4xf32> -> vector<4xi8> (f6E3M2FN)
  llvm.return %res : vector<4xi8>
}

// CHECK-LABEL: @convert_f32x4_to_f6x4_e3m2_rs_relu
llvm.func @convert_f32x4_to_f6x4_e3m2_rs_relu(%src : vector<4xf32>, %rbits : i32) -> vector<4xi8> {
  // CHECK: %{{.*}} = call <4 x i8> @llvm.nvvm.f32x4.to.e3m2x4.rs.relu.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f6x4 %src, %rbits {relu = true} : vector<4xf32> -> vector<4xi8> (f6E3M2FN)
  llvm.return %res : vector<4xi8>
}

// -----

// Test F32x4 -> F4x4 (E2M1) with stochastic rounding (.rs)

// CHECK-LABEL: @convert_f32x4_to_f4x4_e2m1_rs
llvm.func @convert_f32x4_to_f4x4_e2m1_rs(%src : vector<4xf32>, %rbits : i32) -> i16 {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f32x4.to.e2m1x4.rs.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f4x4 %src, %rbits : vector<4xf32> -> i16 (f4E2M1FN)
  llvm.return %res : i16
}

// CHECK-LABEL: @convert_f32x4_to_f4x4_e2m1_rs_relu
llvm.func @convert_f32x4_to_f4x4_e2m1_rs_relu(%src : vector<4xf32>, %rbits : i32) -> i16 {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f32x4.to.e2m1x4.rs.relu.satfinite(<4 x float> %{{.*}}, i32 %{{.*}})
  %res = nvvm.convert.f32x4.to.f4x4 %src, %rbits {relu = true} : vector<4xf32> -> i16 (f4E2M1FN)
  llvm.return %res : i16
}

