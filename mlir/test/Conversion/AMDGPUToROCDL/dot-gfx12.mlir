// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1200 | FileCheck %s

// CHECK-LABEL: @dot_fp8_fp8
func.func @dot_fp8_fp8(%a: vector<4xf8E4M3FN>, %b: vector<4xf8E4M3FN>, %c: f32) -> f32 {
  // CHECK: %[[A:.+]] = llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: %[[B:.+]] = llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: rocdl.dot4.f32.fp8.fp8 %[[A]], %[[B]], %{{.+}} : (i32, i32, f32) -> f32
  %r = amdgpu.dot %a * %b + %c : vector<4xf8E4M3FN>, vector<4xf8E4M3FN>, f32
  func.return %r : f32
}

// CHECK-LABEL: @dot_fp8_bf8
func.func @dot_fp8_bf8(%a: vector<4xf8E4M3FN>, %b: vector<4xf8E5M2>, %c: f32) -> f32 {
  // CHECK: rocdl.dot4.f32.fp8.bf8 %{{.+}}, %{{.+}}, %{{.+}} : (i32, i32, f32) -> f32
  %r = amdgpu.dot %a * %b + %c : vector<4xf8E4M3FN>, vector<4xf8E5M2>, f32
  func.return %r : f32
}

// CHECK-LABEL: @dot_bf8_fp8
func.func @dot_bf8_fp8(%a: vector<4xf8E5M2>, %b: vector<4xf8E4M3FN>, %c: f32) -> f32 {
  // CHECK: rocdl.dot4.f32.bf8.fp8 %{{.+}}, %{{.+}}, %{{.+}} : (i32, i32, f32) -> f32
  %r = amdgpu.dot %a * %b + %c : vector<4xf8E5M2>, vector<4xf8E4M3FN>, f32
  func.return %r : f32
}

// CHECK-LABEL: @dot_bf8_bf8
func.func @dot_bf8_bf8(%a: vector<4xf8E5M2>, %b: vector<4xf8E5M2>, %c: f32) -> f32 {
  // CHECK: rocdl.dot4.f32.bf8.bf8 %{{.+}}, %{{.+}}, %{{.+}} : (i32, i32, f32) -> f32
  %r = amdgpu.dot %a * %b + %c : vector<4xf8E5M2>, vector<4xf8E5M2>, f32
  func.return %r : f32
}
