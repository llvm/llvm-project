// RUN: mlir-opt %s -inline | FileCheck %s

func.func @inner_func_inlinable(%v: f32) -> vector<4xf32> {
  %1 = vector.broadcast %v : f32 to vector<4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: func.func @test_inline(
//  CHECK-NOT:    func.call
//  CHECK-NEXT:   vector.broadcast
func.func @test_inline(%v: f32) -> vector<4xf32> {
  %0 = call @inner_func_inlinable(%v) : (f32) -> vector<4xf32>
  return %0 : vector<4xf32>
}
