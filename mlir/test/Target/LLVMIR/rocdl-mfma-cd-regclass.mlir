// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=ROCDL
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --check-prefix=LLVM --implicit-check-not='"amdgpu-mfma-cd-regclass"="auto"'

llvm.func @automatic(%a: f32, %b: f32, %c: vector<4xf32>) -> vector<4xf32> {
  // ROCDL-LABEL: llvm.func @automatic
  // ROCDL: rocdl.mfma.f32.16x16x4f32 {{.*}}, 0, 0, none :
  // LLVM-LABEL: define <4 x float> @automatic
  // LLVM: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32({{.*}}, i32 0, i32 0, i32 0){{$}}
  %d = rocdl.mfma.f32.16x16x4f32 %a, %b, %c, 0, 0, none : (f32, f32, vector<4xf32>) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

llvm.func @require_agpr(%a: f32, %b: f32, %c: vector<4xf32>) -> vector<4xf32> {
  // ROCDL-LABEL: llvm.func @require_agpr
  // ROCDL: rocdl.mfma.f32.16x16x4f32 {{.*}} cd_regclass = require_agpr :
  // LLVM-LABEL: define <4 x float> @require_agpr
  // LLVM: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32({{.*}}, i32 0, i32 0, i32 0) #[[$AGPR:[0-9]+]]
  %d = rocdl.mfma.f32.16x16x4f32 %a, %b, %c, 0, 0, none cd_regclass = require_agpr : (f32, f32, vector<4xf32>) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

llvm.func @require_vgpr(%a: f32, %b: f32, %c: vector<4xf32>) -> vector<4xf32> {
  // ROCDL-LABEL: llvm.func @require_vgpr
  // ROCDL: rocdl.mfma.f32.16x16x4f32 {{.*}} cd_regclass = require_vgpr :
  // LLVM-LABEL: define <4 x float> @require_vgpr
  // LLVM: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32({{.*}}, i32 0, i32 0, i32 0) #[[$VGPR:[0-9]+]]
  %d = rocdl.mfma.f32.16x16x4f32 %a, %b, %c, 0, 0, none cd_regclass = require_vgpr : (f32, f32, vector<4xf32>) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

llvm.func @explicit_auto(%a: f32, %b: f32, %c: vector<4xf32>) -> vector<4xf32> {
  // ROCDL-LABEL: llvm.func @explicit_auto
  // ROCDL: rocdl.mfma.f32.16x16x4f32 {{.*}}, 0, 0, none :
  // LLVM-LABEL: define <4 x float> @explicit_auto
  // LLVM: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32({{.*}}, i32 0, i32 0, i32 0){{$}}
  %d = rocdl.mfma.f32.16x16x4f32 %a, %b, %c, 0, 0, none cd_regclass = auto : (f32, f32, vector<4xf32>) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

llvm.func @scaled_auto(%a: vector<8xi32>, %b: vector<8xi32>, %c: vector<4xf32>, %sa: i32, %sb: i32) -> vector<4xf32> {
  // ROCDL-LABEL: llvm.func @scaled_auto
  // ROCDL: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}}, fp8_e4m3, fp8_e4m3, 1, {{.*}}, 2, {{.*}} :
  // LLVM-LABEL: define <4 x float> @scaled_auto
  // LLVM: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32({{.*}}, i32 0, i32 0, i32 1, i32 %{{.*}}, i32 2, i32 %{{.*}}){{$}}
  %d = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %a, %b, %c, fp8_e4m3, fp8_e4m3, 1, %sa, 2, %sb : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

llvm.func @scaled_agpr(%a: vector<8xi32>, %b: vector<8xi32>, %c: vector<4xf32>, %sa: i32, %sb: i32) -> vector<4xf32> {
  // ROCDL-LABEL: llvm.func @scaled_agpr
  // ROCDL: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} cd_regclass = require_agpr :
  // LLVM-LABEL: define <4 x float> @scaled_agpr
  // LLVM: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32({{.*}}, i32 0, i32 0, i32 1, i32 %{{.*}}, i32 2, i32 %{{.*}}) #[[$AGPR]]
  %d = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %a, %b, %c, fp8_e4m3, fp8_e4m3, 1, %sa, 2, %sb cd_regclass = require_agpr : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

llvm.func @scaled_vgpr(%a: vector<8xi32>, %b: vector<8xi32>, %c: vector<4xf32>, %sa: i32, %sb: i32) -> vector<4xf32> {
  // ROCDL-LABEL: llvm.func @scaled_vgpr
  // ROCDL: rocdl.mfma.scale.f32.16x16x128.f8f6f4 {{.*}} cd_regclass = require_vgpr :
  // LLVM-LABEL: define <4 x float> @scaled_vgpr
  // LLVM: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32({{.*}}, i32 0, i32 0, i32 1, i32 %{{.*}}, i32 2, i32 %{{.*}}) #[[$VGPR]]
  %d = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %a, %b, %c, fp8_e4m3, fp8_e4m3, 1, %sa, 2, %sb cd_regclass = require_vgpr : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

// LLVM-DAG: attributes #[[$AGPR]] = { {{.*}}"amdgpu-mfma-cd-regclass"="agpr"{{.*}} }
// LLVM-DAG: attributes #[[$VGPR]] = { {{.*}}"amdgpu-mfma-cd-regclass"="vgpr"{{.*}} }
