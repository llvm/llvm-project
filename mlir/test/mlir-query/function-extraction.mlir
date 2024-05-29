// RUN: mlir-query %s -c "m hasOpName(\"arith.mulf\").extract(\"testmul\")" | FileCheck %s

// CHECK: func.func @testmul({{.*}}) -> (f32, f32, f32) {
// CHECK:       %[[MUL0:.*]] = arith.mulf {{.*}} : f32
// CHECK:       %[[MUL1:.*]] = arith.mulf {{.*}}, %[[MUL0]] : f32
// CHECK:       %[[MUL2:.*]] = arith.mulf {{.*}} : f32
// CHECK-NEXT:  return %[[MUL0]], %[[MUL1]], %[[MUL2]] : f32, f32, f32

func.func @mixedOperations(%a: f32, %b: f32, %c: f32) -> f32 {
  %sum0 = arith.addf %a, %b : f32
  %sub0 = arith.subf %sum0, %c : f32
  %mul0 = arith.mulf %a, %sub0 : f32
  %sum1 = arith.addf %b, %c : f32
  %mul1 = arith.mulf %sum1, %mul0 : f32
  %sub2 = arith.subf %mul1, %a : f32
  %sum2 = arith.addf %mul1, %b : f32
  %mul2 = arith.mulf %sub2, %sum2 : f32
  return %mul2 : f32
}
