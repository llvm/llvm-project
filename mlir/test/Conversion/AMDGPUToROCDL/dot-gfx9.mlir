// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx906 | FileCheck %s

// CHECK-LABEL: @dot_fdot2
func.func @dot_fdot2(%a: vector<2xf16>, %b: vector<2xf16>, %c: f32) -> f32 {
  // CHECK: rocdl.fdot2 %{{.+}}, %{{.+}}, %{{.+}} : (vector<2xf16>, vector<2xf16>, f32) -> f32
  %r = amdgpu.dot %a * %b + %c : vector<2xf16>, vector<2xf16>, f32
  func.return %r : f32
}

// CHECK-LABEL: @dot_fdot2_clamp
func.func @dot_fdot2_clamp(%a: vector<2xf16>, %b: vector<2xf16>, %c: f32) -> f32 {
  // CHECK: rocdl.fdot2 %{{.+}}, %{{.+}}, %{{.+}} {clamp = true} : (vector<2xf16>, vector<2xf16>, f32) -> f32
  %r = amdgpu.dot %a * %b + %c {clamp} : vector<2xf16>, vector<2xf16>, f32
  func.return %r : f32
}

// CHECK-LABEL: @dot_sdot2
func.func @dot_sdot2(%a: vector<2xi16>, %b: vector<2xi16>, %c: i32) -> i32 {
  // CHECK: rocdl.sdot2 %{{.+}}, %{{.+}}, %{{.+}} : (vector<2xi16>, vector<2xi16>, i32) -> i32
  %r = amdgpu.dot %a * %b + %c : vector<2xi16>, vector<2xi16>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_udot2_clamp
func.func @dot_udot2_clamp(%a: vector<2xi16>, %b: vector<2xi16>, %c: i32) -> i32 {
  // CHECK: rocdl.udot2 %{{.+}}, %{{.+}}, %{{.+}} {clamp = true} : (vector<2xi16>, vector<2xi16>, i32) -> i32
  %r = amdgpu.dot %a * %b + %c {unsignedA, unsignedB, clamp} : vector<2xi16>, vector<2xi16>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_sdot4
func.func @dot_sdot4(%a: vector<4xi8>, %b: vector<4xi8>, %c: i32) -> i32 {
  // CHECK: %[[A:.+]] = llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: %[[B:.+]] = llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: rocdl.sdot4 %[[A]], %[[B]], %{{.+}} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c : vector<4xi8>, vector<4xi8>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_udot4_clamp
func.func @dot_udot4_clamp(%a: vector<4xi8>, %b: vector<4xi8>, %c: i32) -> i32 {
  // CHECK: rocdl.udot4 %{{.+}}, %{{.+}}, %{{.+}} {clamp = true} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c {unsignedA, unsignedB, clamp} : vector<4xi8>, vector<4xi8>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_sdot8
func.func @dot_sdot8(%a: vector<8xi4>, %b: vector<8xi4>, %c: i32) -> i32 {
  // CHECK: %[[A:.+]] = llvm.bitcast %{{.+}} : vector<8xi4> to i32
  // CHECK: %[[B:.+]] = llvm.bitcast %{{.+}} : vector<8xi4> to i32
  // CHECK: rocdl.sdot8 %[[A]], %[[B]], %{{.+}} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c : vector<8xi4>, vector<8xi4>, i32
  func.return %r : i32
}

// CHECK-LABEL: @dot_udot8
func.func @dot_udot8(%a: vector<8xi4>, %b: vector<8xi4>, %c: i32) -> i32 {
  // CHECK: rocdl.udot8 %{{.+}}, %{{.+}}, %{{.+}} : (i32, i32, i32) -> i32
  %r = amdgpu.dot %a * %b + %c {unsignedA, unsignedB} : vector<8xi4>, vector<8xi4>, i32
  func.return %r : i32
}
