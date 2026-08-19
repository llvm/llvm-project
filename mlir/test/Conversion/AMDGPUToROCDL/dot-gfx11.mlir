// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1100 | FileCheck %s

// CHECK-LABEL: @dot_fdot2_f16_f16
func.func @dot_fdot2_f16_f16(%a: vector<2xf16>, %b: vector<2xf16>, %c: f16) -> f16 {
  // CHECK: rocdl.fdot2.f16.f16 %{{.+}}, %{{.+}}, %{{.+}} : (vector<2xf16>, vector<2xf16>, f16) -> f16
  %r = amdgpu.dot %a * %b + %c : vector<2xf16>, vector<2xf16>, f16
  func.return %r : f16
}

// CHECK-LABEL: @dot_fdot2_bf16_bf16
func.func @dot_fdot2_bf16_bf16(%a: vector<2xbf16>, %b: vector<2xbf16>, %c: bf16) -> bf16 {
  // CHECK: rocdl.fdot2.bf16.bf16 %{{.+}}, %{{.+}}, %{{.+}} : (vector<2xbf16>, vector<2xbf16>, bf16) -> bf16
  %r = amdgpu.dot %a * %b + %c : vector<2xbf16>, vector<2xbf16>, bf16
  func.return %r : bf16
}

// CHECK-LABEL: @dot_fdot2_f32_bf16
func.func @dot_fdot2_f32_bf16(%a: vector<2xbf16>, %b: vector<2xbf16>, %c: f32) -> f32 {
  // CHECK: rocdl.fdot2.f32.bf16 %{{.+}}, %{{.+}}, %{{.+}} : (vector<2xbf16>, vector<2xbf16>, f32) -> f32
  %r = amdgpu.dot %a * %b + %c : vector<2xbf16>, vector<2xbf16>, f32
  func.return %r : f32
}

// Uniform-sign sdot4 still dispatches to the dedicated rocdl.sdot4 (not
// sudot4) on gfx11+. The backend aliases v_dot4_i32_i8 to v_dot4_i32_iu8.
// CHECK-LABEL: @dot_sdot4_gfx11_uniform_sign
func.func @dot_sdot4_gfx11_uniform_sign(%a: vector<4xi8>, %b: vector<4xi8>, %c: i32) -> i32 {
  // CHECK: rocdl.sdot4 %{{.+}}, %{{.+}}, %{{.+}} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c : vector<4xi8>, vector<4xi8>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_sudot4_signA_unsignedB
func.func @dot_sudot4_signA_unsignedB(%a: vector<4xi8>, %b: vector<4xi8>, %c: i32) -> i32 {
  // CHECK: rocdl.sudot4 %{{.+}}, %{{.+}}, %{{.+}} {signA = true} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c {unsignedB} : vector<4xi8>, vector<4xi8>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_sudot4_unsignedA_signB_clamp
func.func @dot_sudot4_unsignedA_signB_clamp(%a: vector<4xi8>, %b: vector<4xi8>, %c: i32) -> i32 {
  // CHECK: rocdl.sudot4 %{{.+}}, %{{.+}}, %{{.+}} {clamp = true, signB = true} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c {unsignedA, clamp} : vector<4xi8>, vector<4xi8>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_sudot8
func.func @dot_sudot8(%a: vector<8xi4>, %b: vector<8xi4>, %c: i32) -> i32 {
  // CHECK: rocdl.sudot8 %{{.+}}, %{{.+}}, %{{.+}} {signA = true} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c {unsignedB} : vector<8xi4>, vector<8xi4>, i32
  func.return %r : i32
}
