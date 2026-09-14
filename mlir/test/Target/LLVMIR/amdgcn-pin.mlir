// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The register-pinning intrinsics are reachable from MLIR via llvm.call_intrinsic
// (the mechanism DSLs such as FlyDSL use); the overload is mangled from the
// operand type. Requires an LLVM that defines llvm.amdgcn.pin.*.

// CHECK-LABEL: define <2 x i32> @pin_agpr
llvm.func @pin_agpr(%v: vector<2xi32>) -> vector<2xi32> {
  %r = llvm.mlir.constant(8 : i32) : i32
  // CHECK: call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %{{[0-9]+}}, i32 8)
  %p = llvm.call_intrinsic "llvm.amdgcn.pin.agpr"(%v, %r) : (vector<2xi32>, i32) -> vector<2xi32>
  llvm.return %p : vector<2xi32>
}

// CHECK-LABEL: define <4 x float> @pin_vgpr
llvm.func @pin_vgpr(%v: vector<4xf32>) -> vector<4xf32> {
  %r = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %{{[0-9]+}}, i32 0)
  %p = llvm.call_intrinsic "llvm.amdgcn.pin.vgpr"(%v, %r) : (vector<4xf32>, i32) -> vector<4xf32>
  llvm.return %p : vector<4xf32>
}
