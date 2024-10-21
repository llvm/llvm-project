// RUN: mlir-opt --memref-emulate-wide-int="widest-int-supported=32" %s \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s

// Expect no conversions, i32 is supported.
// CHECK-LABEL: func @memref_i32
// CHECK:         [[M:%.+]] = memref.alloc() : memref<4xi32, 1>
// CHECK-NEXT:    [[V:%.+]] = memref.load [[M]][{{%.+}}] : memref<4xi32, 1>
// CHECK-NEXT:    memref.store {{%.+}}, [[M]][{{%.+}}] : memref<4xi32, 1>
// CHECK-NEXT:    return
func.func @memref_i32() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %m = memref.alloc() : memref<4xi32, 1>
    %v = memref.load %m[%c0] : memref<4xi32, 1>
    memref.store %c1, %m[%c0] : memref<4xi32, 1>
    return
}

// -----

// Expect no conversions, f64 is not an integer type.
// CHECK-LABEL: func @memref_f32
// CHECK:         [[M:%.+]] = memref.alloc() : memref<4xf32, 1>
// CHECK-NEXT:    [[V:%.+]] = memref.load [[M]][{{%.+}}] : memref<4xf32, 1>
// CHECK-NEXT:    memref.store {{%.+}}, [[M]][{{%.+}}] : memref<4xf32, 1>
// CHECK-NEXT:    return
func.func @memref_f32() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1.0 : f32
    %m = memref.alloc() : memref<4xf32, 1>
    %v = memref.load %m[%c0] : memref<4xf32, 1>
    memref.store %c1, %m[%c0] : memref<4xf32, 1>
    return
}

// -----

// CHECK-LABEL: func @alloc_load_store_i64
// CHECK:         [[C1:%.+]] = arith.constant dense<[1, 0]> : vector<2xi32>
// CHECK-NEXT:    [[M:%.+]]  = memref.alloc() : memref<4xvector<2xi32>, 1>
// CHECK-NEXT:    [[V:%.+]]  = memref.load [[M]][{{%.+}}] : memref<4xvector<2xi32>, 1>
// CHECK-NEXT:    memref.store [[C1]], [[M]][{{%.+}}] : memref<4xvector<2xi32>, 1>
// CHECK-NEXT:    return
func.func @alloc_load_store_i64() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i64
    %m = memref.alloc() : memref<4xi64, 1>
    %v = memref.load %m[%c0] : memref<4xi64, 1>
    memref.store %c1, %m[%c0] : memref<4xi64, 1>
    return
}

// -----

// CHECK-LABEL: func @alloc_load_store_i64_nontemporal
// CHECK:         [[C1:%.+]] = arith.constant dense<[1, 0]> : vector<2xi32>
// CHECK-NEXT:    [[M:%.+]]  = memref.alloc() : memref<4xvector<2xi32>, 1>
// CHECK-NEXT:    [[V:%.+]]  = memref.load [[M]][{{%.+}}] {nontemporal = true} : memref<4xvector<2xi32>, 1>
// CHECK-NEXT:    memref.store [[C1]], [[M]][{{%.+}}] {nontemporal = true} : memref<4xvector<2xi32>, 1>
// CHECK-NEXT:    return
func.func @alloc_load_store_i64_nontemporal() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i64
    %m = memref.alloc() : memref<4xi64, 1>
    %v = memref.load %m[%c0] {nontemporal = true} : memref<4xi64, 1>
    memref.store %c1, %m[%c0] {nontemporal = true} : memref<4xi64, 1>
    return
}

// -----

// Make sure we do not crash on unsupported types.
func.func @alloc_i128() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloc' that was explicitly marked illegal}}
  %m = memref.alloc() : memref<4xi128, 1>
  return
}

// -----

func.func @load_i128(%m: memref<4xi128, 1>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{failed to legalize operation 'memref.load' that was explicitly marked illegal}}
  %v = memref.load %m[%c0] : memref<4xi128, 1>
  return
}

// -----

func.func @store_i128(%c1: i128, %m: memref<4xi128, 1>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{failed to legalize operation 'memref.store' that was explicitly marked illegal}}
  memref.store %c1, %m[%c0] : memref<4xi128, 1>
  return
}
