// RUN: mlir-opt %s --amdgpu-vector-reduction-to-dot=chipset=gfx950 --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @i16_signed_ext
func.func @i16_signed_ext(%lhs: vector<4xi16>, %rhs: vector<4xi16>,
                          %acc: i32) -> i32 {
  %lhs_ext = arith.extsi %lhs : vector<4xi16> to vector<4xi32>
  %rhs_ext = arith.extsi %rhs : vector<4xi16> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK: amdgpu.dot {{.*}} : vector<2xi16>, vector<2xi16>, i32
// CHECK: amdgpu.dot {{.*}} : vector<2xi16>, vector<2xi16>, i32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @i16_unsigned_ext
func.func @i16_unsigned_ext(%lhs: vector<4xi16>, %rhs: vector<4xi16>,
                            %acc: i32) -> i32 {
  %lhs_ext = arith.extui %lhs : vector<4xi16> to vector<4xi32>
  %rhs_ext = arith.extui %rhs : vector<4xi16> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK: amdgpu.dot {{.*}} {unsignedA, unsignedB} : vector<2xi16>, vector<2xi16>, i32
// CHECK: amdgpu.dot {{.*}} {unsignedA, unsignedB} : vector<2xi16>, vector<2xi16>, i32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_f16_direct_target_disabled
func.func @nofold_f16_direct_target_disabled(%lhs: vector<4xf16>,
                                             %rhs: vector<4xf16>,
                                             %acc: f16) -> f16 {
  %mul = arith.mulf %lhs, %rhs fastmath<contract,reassoc> : vector<4xf16>
  %red = vector.reduction <add>, %mul, %acc fastmath<contract,reassoc>
    : vector<4xf16> into f16
  return %red : f16
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction
