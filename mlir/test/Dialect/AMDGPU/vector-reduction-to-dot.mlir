// RUN: mlir-opt %s --amdgpu-vector-reduction-to-dot=chipset=gfx1200 --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @f16_extf_extract
// CHECK-SAME: %[[LHS:.+]]: vector<64x8xf16>
// CHECK-SAME: %[[RHS:.+]]: vector<8xf16>
// CHECK-SAME: %[[ACC:.+]]: f32
func.func @f16_extf_extract(%lhs: vector<64x8xf16>, %rhs: vector<8xf16>,
                            %acc: f32) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<64x8xf16> to vector<64x8xf32>
  %rhs_ext = arith.extf %rhs : vector<8xf16> to vector<8xf32>
  %row = vector.extract %lhs_ext[0] : vector<8xf32> from vector<64x8xf32>
  %mul = arith.mulf %row, %rhs_ext fastmath<contract,reassoc> : vector<8xf32>
  %red = vector.reduction <add>, %mul, %acc fastmath<contract,reassoc>
    : vector<8xf32> into f32
  return %red : f32
}
// CHECK: %[[ROW:.+]] = vector.extract %[[LHS]][0] : vector<8xf16> from vector<64x8xf16>
// CHECK: %[[LHS0:.+]] = vector.extract_strided_slice %[[ROW]] {offsets = [0], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
// CHECK: %[[RHS0:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
// CHECK: %[[DOT0:.+]] = amdgpu.dot %[[LHS0]] * %[[RHS0]] + %[[ACC]] : vector<2xf16>, vector<2xf16>, f32
// CHECK: %[[LHS1:.+]] = vector.extract_strided_slice %[[ROW]] {offsets = [2], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
// CHECK: %[[RHS1:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [2], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
// CHECK: %[[DOT1:.+]] = amdgpu.dot %[[LHS1]] * %[[RHS1]] + %[[DOT0]] : vector<2xf16>, vector<2xf16>, f32
// CHECK: %[[DOT2:.+]] = amdgpu.dot {{.*}} + %[[DOT1]] : vector<2xf16>, vector<2xf16>, f32
// CHECK: %[[DOT3:.+]] = amdgpu.dot {{.*}} + %[[DOT2]] : vector<2xf16>, vector<2xf16>, f32
// CHECK-NOT: vector.reduction
// CHECK: return %[[DOT3]] : f32

// -----

// CHECK-LABEL: func.func @f16_direct_accumulator
func.func @f16_direct_accumulator(%lhs: vector<4xf16>, %rhs: vector<4xf16>,
                                  %acc: f16) -> f16 {
  %mul = arith.mulf %lhs, %rhs fastmath<contract,reassoc> : vector<4xf16>
  %red = vector.reduction <add>, %mul, %acc fastmath<contract,reassoc>
    : vector<4xf16> into f16
  return %red : f16
}
// CHECK: amdgpu.dot {{.*}} : vector<2xf16>, vector<2xf16>, f16
// CHECK: amdgpu.dot {{.*}} : vector<2xf16>, vector<2xf16>, f16
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @bf16_direct_accumulator
func.func @bf16_direct_accumulator(%lhs: vector<4xbf16>,
                                   %rhs: vector<4xbf16>,
                                   %acc: bf16) -> bf16 {
  %mul = arith.mulf %lhs, %rhs fastmath<contract,reassoc> : vector<4xbf16>
  %red = vector.reduction <add>, %mul, %acc fastmath<contract,reassoc>
    : vector<4xbf16> into bf16
  return %red : bf16
}
// CHECK: amdgpu.dot {{.*}} : vector<2xbf16>, vector<2xbf16>, bf16
// CHECK: amdgpu.dot {{.*}} : vector<2xbf16>, vector<2xbf16>, bf16
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @bf16_extf
func.func @bf16_extf(%lhs: vector<4xbf16>, %rhs: vector<4xbf16>,
                     %acc: f32) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<4xbf16> to vector<4xf32>
  %rhs_ext = arith.extf %rhs : vector<4xbf16> to vector<4xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext fastmath<contract,reassoc>
    : vector<4xf32>
  %red = vector.reduction <add>, %mul, %acc fastmath<contract,reassoc>
    : vector<4xf32> into f32
  return %red : f32
}
// CHECK: amdgpu.dot {{.*}} : vector<2xbf16>, vector<2xbf16>, f32
// CHECK: amdgpu.dot {{.*}} : vector<2xbf16>, vector<2xbf16>, f32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @fp8_mixed_extf
func.func @fp8_mixed_extf(%lhs: vector<4xf8E4M3FN>,
                          %rhs: vector<4xf8E5M2>) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<4xf8E4M3FN> to vector<4xf32>
  %rhs_ext = arith.extf %rhs : vector<4xf8E5M2> to vector<4xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext fastmath<contract,reassoc>
    : vector<4xf32>
  %red = vector.reduction <add>, %mul fastmath<contract,reassoc,nsz>
    : vector<4xf32> into f32
  return %red : f32
}
// CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: amdgpu.dot {{.*}} + %[[ZERO]] : vector<4xf8E4M3FN>, vector<4xf8E5M2>, f32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_i16_signed_target_disabled
func.func @nofold_i16_signed_target_disabled(%lhs: vector<4xi16>,
                                             %rhs: vector<4xi16>,
                                             %acc: i32) -> i32 {
  %lhs_ext = arith.extsi %lhs : vector<4xi16> to vector<4xi32>
  %rhs_ext = arith.extsi %rhs : vector<4xi16> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_i16_unsigned_target_disabled
func.func @nofold_i16_unsigned_target_disabled(%lhs: vector<4xi16>,
                                               %rhs: vector<4xi16>,
                                               %acc: i32) -> i32 {
  %lhs_ext = arith.extui %lhs : vector<4xi16> to vector<4xi32>
  %rhs_ext = arith.extui %rhs : vector<4xi16> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @i8_mixed_sign_ext
func.func @i8_mixed_sign_ext(%lhs: vector<8xi8>, %rhs: vector<8xi8>,
                             %acc: i32) -> i32 {
  %lhs_ext = arith.extsi %lhs : vector<8xi8> to vector<8xi32>
  %rhs_ext = arith.extui %rhs : vector<8xi8> to vector<8xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<8xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<8xi32> into i32
  return %red : i32
}
// CHECK: amdgpu.dot {{.*}} {unsignedB} : vector<4xi8>, vector<4xi8>, i32
// CHECK: amdgpu.dot {{.*}} {unsignedB} : vector<4xi8>, vector<4xi8>, i32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @i8_unsigned_a_ext
func.func @i8_unsigned_a_ext(%lhs: vector<4xi8>, %rhs: vector<4xi8>,
                             %acc: i32) -> i32 {
  %lhs_ext = arith.extui %lhs : vector<4xi8> to vector<4xi32>
  %rhs_ext = arith.extsi %rhs : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK: amdgpu.dot {{.*}} {unsignedA} : vector<4xi8>, vector<4xi8>, i32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @i8_unsigned_ab_ext
func.func @i8_unsigned_ab_ext(%lhs: vector<4xi8>, %rhs: vector<4xi8>,
                              %acc: i32) -> i32 {
  %lhs_ext = arith.extui %lhs : vector<4xi8> to vector<4xi32>
  %rhs_ext = arith.extui %rhs : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK: amdgpu.dot {{.*}} {unsignedA, unsignedB} : vector<4xi8>, vector<4xi8>, i32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @i4_unsigned_ext
func.func @i4_unsigned_ext(%lhs: vector<8xi4>, %rhs: vector<8xi4>,
                           %acc: i32) -> i32 {
  %lhs_ext = arith.extui %lhs : vector<8xi4> to vector<8xi32>
  %rhs_ext = arith.extui %rhs : vector<8xi4> to vector<8xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<8xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<8xi32> into i32
  return %red : i32
}
// CHECK: amdgpu.dot {{.*}} {unsignedA, unsignedB} : vector<8xi4>, vector<8xi4>, i32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @i4_mixed_sign_ext
func.func @i4_mixed_sign_ext(%lhs: vector<8xi4>, %rhs: vector<8xi4>,
                             %acc: i32) -> i32 {
  %lhs_ext = arith.extsi %lhs : vector<8xi4> to vector<8xi32>
  %rhs_ext = arith.extui %rhs : vector<8xi4> to vector<8xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<8xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<8xi32> into i32
  return %red : i32
}
// CHECK: amdgpu.dot {{.*}} {unsignedB} : vector<8xi4>, vector<8xi4>, i32
// CHECK-NOT: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_missing_fastmath
func.func @nofold_missing_fastmath(%lhs: vector<4xf16>,
                                   %rhs: vector<4xf16>) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<4xf16> to vector<4xf32>
  %rhs_ext = arith.extf %rhs : vector<4xf16> to vector<4xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext : vector<4xf32>
  %red = vector.reduction <add>, %mul : vector<4xf32> into f32
  return %red : f32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_mul_missing_contract
func.func @nofold_mul_missing_contract(%lhs: vector<4xf16>,
                                       %rhs: vector<4xf16>) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<4xf16> to vector<4xf32>
  %rhs_ext = arith.extf %rhs : vector<4xf16> to vector<4xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext fastmath<reassoc> : vector<4xf32>
  %red = vector.reduction <add>, %mul fastmath<contract,reassoc,nsz>
    : vector<4xf32> into f32
  return %red : f32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_reduction_missing_contract
func.func @nofold_reduction_missing_contract(%lhs: vector<4xf16>,
                                             %rhs: vector<4xf16>) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<4xf16> to vector<4xf32>
  %rhs_ext = arith.extf %rhs : vector<4xf16> to vector<4xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext fastmath<contract,reassoc>
    : vector<4xf32>
  %red = vector.reduction <add>, %mul fastmath<reassoc,nsz>
    : vector<4xf32> into f32
  return %red : f32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_reduction_missing_reassoc
func.func @nofold_reduction_missing_reassoc(%lhs: vector<4xf16>,
                                            %rhs: vector<4xf16>) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<4xf16> to vector<4xf32>
  %rhs_ext = arith.extf %rhs : vector<4xf16> to vector<4xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext fastmath<contract> : vector<4xf32>
  %red = vector.reduction <add>, %mul fastmath<contract,nsz>
    : vector<4xf32> into f32
  return %red : f32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_no_acc_missing_nsz
func.func @nofold_no_acc_missing_nsz(%lhs: vector<4xf16>,
                                     %rhs: vector<4xf16>) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<4xf16> to vector<4xf32>
  %rhs_ext = arith.extf %rhs : vector<4xf16> to vector<4xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext fastmath<contract> : vector<4xf32>
  %red = vector.reduction <add>, %mul fastmath<contract,reassoc>
    : vector<4xf32> into f32
  return %red : f32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_non_multiple
func.func @nofold_non_multiple(%lhs: vector<3xf16>,
                               %rhs: vector<3xf16>) -> f32 {
  %lhs_ext = arith.extf %lhs : vector<3xf16> to vector<3xf32>
  %rhs_ext = arith.extf %rhs : vector<3xf16> to vector<3xf32>
  %mul = arith.mulf %lhs_ext, %rhs_ext fastmath<contract,reassoc>
    : vector<3xf32>
  %red = vector.reduction <add>, %mul fastmath<contract,reassoc,nsz>
    : vector<3xf32> into f32
  return %red : f32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_i16_mixed_sign
func.func @nofold_i16_mixed_sign(%lhs: vector<4xi16>, %rhs: vector<4xi16>,
                                 %acc: i32) -> i32 {
  %lhs_ext = arith.extsi %lhs : vector<4xi16> to vector<4xi32>
  %rhs_ext = arith.extui %rhs : vector<4xi16> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_integer_overflow_flags
func.func @nofold_integer_overflow_flags(%lhs: vector<4xi8>,
                                         %rhs: vector<4xi8>,
                                         %acc: i32) -> i32 {
  %lhs_ext = arith.extui %lhs : vector<4xi8> to vector<4xi32>
  %rhs_ext = arith.extui %rhs : vector<4xi8> to vector<4xi32>
  %mul = arith.muli %lhs_ext, %rhs_ext overflow<nsw> : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction

// -----

// CHECK-LABEL: func.func @nofold_direct_i32
func.func @nofold_direct_i32(%lhs: vector<4xi32>, %rhs: vector<4xi32>,
                             %acc: i32) -> i32 {
  %mul = arith.muli %lhs, %rhs : vector<4xi32>
  %red = vector.reduction <add>, %mul, %acc : vector<4xi32> into i32
  return %red : i32
}
// CHECK-NOT: amdgpu.dot
// CHECK: vector.reduction
