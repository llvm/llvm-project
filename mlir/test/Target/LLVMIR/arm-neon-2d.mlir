// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(arm-neon-2d-to-intr))" %s | FileCheck %s

// CHECK-LABEL: arm_neon_sdot2d_4x4_i8i8
func.func @arm_neon_sdot2d_4x4_i8i8(%a: vector<4xi32>, %b: vector<4x4xi8>, %c: vector<4x4xi8>) -> vector<4xi32> {
  // CHECK: arm_neon.intr.sdot %{{.*}}, %{{.*}}, %{{.*}} : vector<16xi8>, vector<16xi8> to vector<4xi32>
  // CHECK-NEXT: return %{{.*}} : vector<4xi32>
  %0 = arm_neon.2d.sdot %a, %b, %c : vector<4x4xi8>, vector<4x4xi8> to vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: arm_neon_sdot2d_2x4_i8i8
func.func @arm_neon_sdot2d_2x4_i8i8(%a: vector<2xi32>, %b: vector<2x4xi8>, %c: vector<2x4xi8>) -> vector<2xi32> {
  // CHECK: arm_neon.intr.sdot %{{.*}}, %{{.*}}, %{{.*}} : vector<8xi8>, vector<8xi8> to vector<2xi32>
  // CHECK-NEXT: return %{{.*}} : vector<2xi32>
  %0 = arm_neon.2d.sdot %a, %b, %c : vector<2x4xi8>, vector<2x4xi8> to vector<2xi32>
  return %0 : vector<2xi32>
}
