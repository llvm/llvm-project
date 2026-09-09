// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %inf = arith.constant 0x7F800000 : f32
  %ninf = arith.constant 0xFF800000 : f32
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>

  // Ordinary negative data. A neutral value anywhere above the active lanes,
  // such as a negative subnormal, would be returned instead of the maximum.
  %v0 = arith.constant dense<[-5.0, -7.0, -5.0, -9.0]> : vector<4xf32>
  %0 = vector.mask %mask {
    vector.reduction <maximumf>, %v0 : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %0 : f32
  // CHECK: -5

  // Mirror image for the minimum.
  %v1 = arith.constant dense<[5.0, 7.0, 5.0, 9.0]> : vector<4xf32>
  %1 = vector.mask %mask {
    vector.reduction <minimumf>, %v1 : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %1 : f32
  // CHECK: 5

  // Infinite active lanes: the largest finite value would win over them.
  %v2 = vector.broadcast %ninf : f32 to vector<4xf32>
  %2 = vector.mask %mask {
    vector.reduction <maximumf>, %v2 : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %2 : f32
  // CHECK: -inf

  %v3 = vector.broadcast %inf : f32 to vector<4xf32>
  %3 = vector.mask %mask {
    vector.reduction <minimumf>, %v3 : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %3 : f32
  // CHECK: inf

  // No active lane: the result is the identity of the reduction.
  %nomask = arith.constant dense<false> : vector<4xi1>
  %4 = vector.mask %nomask {
    vector.reduction <maximumf>, %v0 : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %4 : f32
  // CHECK: -inf

  %5 = vector.mask %nomask {
    vector.reduction <minimumf>, %v1 : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %5 : f32
  // CHECK: inf

  // An accumulator combines with the masked reduction.
  %acc = arith.constant -3.0 : f32
  %6 = vector.mask %mask {
    vector.reduction <maximumf>, %v0, %acc : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %6 : f32
  // CHECK: -3

  // Under `ninf` the neutral value is the largest finite value instead. It is
  // still neutral for the finite inputs the flag restricts the reduction to.
  %7 = vector.mask %mask {
    vector.reduction <maximumf>, %v0 fastmath<ninf> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %7 : f32
  // CHECK: -5

  %8 = vector.mask %mask {
    vector.reduction <minimumf>, %v1 fastmath<ninf> : vector<4xf32> into f32
  } : vector<4xi1> -> f32
  vector.print %8 : f32
  // CHECK: 5

  return
}
