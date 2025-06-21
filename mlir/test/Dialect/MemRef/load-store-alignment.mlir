// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @test_load_store_alignment
// CHECK: memref.load {{.*}} {alignment = 16 : i64}
// CHECK: memref.store {{.*}} {alignment = 16 : i64}
func.func @test_load_store_alignment(%memref: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %val = memref.load %memref[%c0] { alignment = 16 } : memref<4xi32>
  memref.store %val, %memref[%c0] { alignment = 16 } : memref<4xi32>
  return
}

// -----

func.func @test_invalid_negative_load_alignment(%memref: memref<4xi32>) {
  // expected-error @+1 {{custom op 'memref.load' 'memref.load' op attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %val = memref.load %memref[%c0] { alignment = -1 } : memref<4xi32>
  return
}

// -----

func.func @test_invalid_non_power_of_2_store_alignment(%memref: memref<4xi32>, %val: i32) {
  // expected-error @+1 {{custom op 'memref.store' 'memref.store' op attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  memref.store %val, %memref[%c0] { alignment = 1 } : memref<4xi32>
  return
}
