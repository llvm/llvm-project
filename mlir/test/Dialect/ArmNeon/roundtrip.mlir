// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

// CHECK-LABEL: arm_neon_smull
func @arm_neon_smull(%a: vector<8xi8>, %b: vector<8xi8>)
    -> (vector<8xi16>, vector<4xi32>, vector<2xi64>) {
  // CHECK: arm_neon.intr.smull {{.*}}: vector<8xi8> to vector<8xi16>
  %0 = arm_neon.intr.smull %a, %b : vector<8xi8> to vector<8xi16>
  %00 = vector.extract_strided_slice %0 {offsets = [3], sizes = [4], strides = [1]}:
    vector<8xi16> to vector<4xi16>

  // CHECK: arm_neon.intr.smull {{.*}}: vector<4xi16> to vector<4xi32>
  %1 = arm_neon.intr.smull %00, %00 : vector<4xi16> to vector<4xi32>
  %11 = vector.extract_strided_slice %1 {offsets = [1], sizes = [2], strides = [1]}:
    vector<4xi32> to vector<2xi32>

  // CHECK: arm_neon.intr.smull {{.*}}: vector<2xi32> to vector<2xi64>
  %2 = arm_neon.intr.smull %11, %11 : vector<2xi32> to vector<2xi64>

  return %0, %1, %2 : vector<8xi16>, vector<4xi32>, vector<2xi64>
}

// CHECK-LABEL: arm_neon_sdot
func @arm_neon_sdot(%a: vector<2xi32>, %b: vector<8xi8>, %c: vector<8xi8>) -> vector<2xi32> {
  // CHECK: arm_neon.intr.sdot {{.*}}: vector<8xi8>, vector<8xi8> to vector<2xi32>
  %0 = arm_neon.intr.sdot %a, %b, %c : vector<8xi8>, vector<8xi8> to vector<2xi32>
  return %0 : vector<2xi32>
}
