// RUN: mlir-opt -int-range-optimizations -canonicalize %s | FileCheck %s


// CHECK-LABEL: func @constant_vec
// CHECK: test.reflect_bounds {smax = 7 : index, smin = 0 : index, umax = 7 : index, umin = 0 : index}
func.func @constant_vec() -> vector<8xindex> {
  %0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
  %1 = test.reflect_bounds %0 : vector<8xindex>
  func.return %1 : vector<8xindex>
}

// CHECK-LABEL: func @constant_splat
// CHECK: test.reflect_bounds {smax = 3 : si32, smin = 3 : si32, umax = 3 : ui32, umin = 3 : ui32}
func.func @constant_splat() -> vector<8xi32> {
  %0 = arith.constant dense<3> : vector<8xi32>
  %1 = test.reflect_bounds %0 : vector<8xi32>
  func.return %1 : vector<8xi32>
}

// CHECK-LABEL: func @float_constant_splat
// Don't crash on splat floats.
func.func @float_constant_splat() -> vector<8xf32> {
  %0 = arith.constant dense<3.0> : vector<8xf32>
  func.return %0: vector<8xf32>
}

// CHECK-LABEL: func @vector_splat
// CHECK: test.reflect_bounds {smax = 5 : index, smin = 4 : index, umax = 5 : index, umin = 4 : index}
func.func @vector_splat() -> vector<4xindex> {
  %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : index
  %1 = vector.broadcast %0 : index to vector<4xindex>
  %2 = test.reflect_bounds %1 : vector<4xindex>
  func.return %2 : vector<4xindex>
}

// CHECK-LABEL: func @vector_broadcast
// CHECK: test.reflect_bounds {smax = 5 : index, smin = 4 : index, umax = 5 : index, umin = 4 : index}
func.func @vector_broadcast() -> vector<4x16xindex> {
  %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : vector<16xindex>
  %1 = vector.broadcast %0 : vector<16xindex> to vector<4x16xindex>
  %2 = test.reflect_bounds %1 : vector<4x16xindex>
  func.return %2 : vector<4x16xindex>
}

// CHECK-LABEL: func @vector_shape_cast
// CHECK: test.reflect_bounds {smax = 5 : index, smin = 4 : index, umax = 5 : index, umin = 4 : index}
func.func @vector_shape_cast() -> vector<4x4xindex> {
  %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : vector<16xindex>
  %1 = vector.shape_cast %0 : vector<16xindex> to vector<4x4xindex>
  %2 = test.reflect_bounds %1 : vector<4x4xindex>
  func.return %2 : vector<4x4xindex>
}

// CHECK-LABEL: func @vector_transpose
// CHECK: test.reflect_bounds {smax = 8 : index, smin = 7 : index, umax = 8 : index, umin = 7 : index}
func.func @vector_transpose() -> vector<2x4xindex> {
  %0 = test.with_bounds { smax = 8 : index, smin = 7 : index, umax = 8 : index, umin = 7 : index } : vector<4x2xindex>
  %1 = vector.transpose %0, [1, 0] : vector<4x2xindex> to vector<2x4xindex>
  %2 = test.reflect_bounds %1 : vector<2x4xindex>
  func.return %2 : vector<2x4xindex>
}

// CHECK-LABEL: func @vector_extract
// CHECK: test.reflect_bounds {smax = 6 : index, smin = 5 : index, umax = 6 : index, umin = 5 : index}
func.func @vector_extract() -> index {
  %0 = test.with_bounds { umin = 5 : index, umax = 6 : index, smin = 5 : index, smax = 6 : index } : vector<4xindex>
  %1 = vector.extract %0[0] : index from vector<4xindex>
  %2 = test.reflect_bounds %1 : index
  func.return %2 : index
}

// CHECK-LABEL: func @vector_add
// CHECK: test.reflect_bounds {smax = 12 : index, smin = 10 : index, umax = 12 : index, umin = 10 : index}
func.func @vector_add() -> vector<4xindex> {
  %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : vector<4xindex>
  %1 = test.with_bounds { umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index } : vector<4xindex>
  %2 = arith.addi %0, %1 : vector<4xindex>
  %3 = test.reflect_bounds %2 : vector<4xindex>
  func.return %3 : vector<4xindex>
}

// CHECK-LABEL: func @vector_insert
// CHECK: test.reflect_bounds {smax = 8 : index, smin = 5 : index, umax = 8 : index, umin = 5 : index}
func.func @vector_insert() -> vector<4xindex> {
  %0 = test.with_bounds { umin = 5 : index, umax = 7 : index, smin = 5 : index, smax = 7 : index } : vector<4xindex>
  %1 = test.with_bounds { umin = 6 : index, umax = 8 : index, smin = 6 : index, smax = 8 : index } : index
  %2 = vector.insert %1, %0[0] : index into vector<4xindex>
  %3 = test.reflect_bounds %2 : vector<4xindex>
  func.return %3 : vector<4xindex>
}

// CHECK-LABEL: func @test_loaded_vector_extract
// No bounds
// CHECK: test.reflect_bounds {smax = 2147483647 : si32, smin = -2147483648 : si32, umax = 4294967295 : ui32, umin = 0 : ui32} %{{.*}} : i32
func.func @test_loaded_vector_extract(%memref : memref<16xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %v = vector.load %memref[%c0] : memref<16xi32>, vector<4xi32>
  %e = vector.extract %v[0] : i32 from vector<4xi32>
  %bounds = test.reflect_bounds %e : i32
  func.return %bounds : i32
}

// CHECK-LABEL: func @test_vector_extsi
// CHECK: test.reflect_bounds {smax = 5 : si32, smin = 1 : si32, umax = 5 : ui32, umin = 1 : ui32}
func.func @test_vector_extsi() -> vector<2xi32> {
  %0 = test.with_bounds {smax = 5 : si8, smin = 1 : si8, umax = 5 : ui8, umin = 1 : ui8 } : vector<2xi8>
  %1 = arith.extsi %0 : vector<2xi8> to vector<2xi32>
  %2 = test.reflect_bounds %1 : vector<2xi32>
  func.return %2 : vector<2xi32>
}

// CHECK-LABEL: func @vector_step
// CHECK: test.reflect_bounds {smax = 7 : index, smin = 0 : index, umax = 7 : index, umin = 0 : index}
func.func @vector_step() -> vector<8xindex> {
  %0 = vector.step : vector<8xindex>
  %1 = test.reflect_bounds %0 : vector<8xindex>
  func.return %1 : vector<8xindex>
}
