// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @test_masked_load_store_alignment
// CHECK: vector.maskedload {{.*}} {alignment = 16 : i64}
// CHECK: vector.maskedstore {{.*}} {alignment = 16 : i64}
func.func @test_masked_load_store_alignment(%memref: memref<4xi32>, %mask: vector<4xi1>, %passthru: vector<4xi32>) {
  %c0 = arith.constant 0 : index
  %val = vector.maskedload %memref[%c0], %mask, %passthru { alignment = 16 } : memref<4xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
  vector.maskedstore %memref[%c0], %mask,  %val { alignment = 16 } : memref<4xi32>, vector<4xi1>, vector<4xi32>
  return
}

// -----

// CHECK-LABEL: func @test_load_store_alignment
// CHECK: vector.load {{.*}} {alignment = 16 : i64}
// CHECK: vector.store {{.*}} {alignment = 16 : i64}
func.func @test_load_store_alignment(%memref: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %val = vector.load %memref[%c0] { alignment = 16 } : memref<4xi32>, vector<4xi32>
  vector.store %val, %memref[%c0] { alignment = 16 } : memref<4xi32>, vector<4xi32>
  return
}

// -----

func.func @test_invalid_negative_load_alignment(%memref: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{custom op 'vector.load' 'vector.load' op attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %val = vector.load %memref[%c0] { alignment = -1 } : memref<4xi32>, vector<4xi32>
  return
}

// -----

func.func @test_invalid_non_power_of_2_store_alignment(%memref: memref<4xi32>, %val: vector<4xi32>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{custom op 'vector.store' 'vector.store' op attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  vector.store %val, %memref[%c0] { alignment = 3 } : memref<4xi32>, vector<4xi32>
  return
}
