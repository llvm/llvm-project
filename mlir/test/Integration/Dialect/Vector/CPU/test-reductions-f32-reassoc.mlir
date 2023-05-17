// RUN: mlir-opt %s -convert-scf-to-cf \
// RUN:             -convert-vector-to-llvm='reassociate-fp-reductions' \
// RUN:             -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  // Construct test vector, numerically very stable.
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %v0 = vector.broadcast %f1 : f32 to vector<64xf32>
  %v1 = vector.insert %f2, %v0[11] : f32 into vector<64xf32>
  %v2 = vector.insert %f3, %v1[52] : f32 into vector<64xf32>
  vector.print %v2 : vector<64xf32>
  //
  // test vector:
  //
  // CHECK: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )

  // Various vector reductions. Not full functional unit tests, but
  // a simple integration test to see if the code runs end-to-end.
  %0 = vector.reduction <add>, %v2 : vector<64xf32> into f32
  vector.print %0 : f32
  // CHECK: 67
  %1 = vector.reduction <mul>, %v2 : vector<64xf32> into f32
  vector.print %1 : f32
  // CHECK: 6
  %2 = vector.reduction <minf>, %v2 : vector<64xf32> into f32
  vector.print %2 : f32
  // CHECK: 1
  %3 = vector.reduction <maxf>, %v2 : vector<64xf32> into f32
  vector.print %3 : f32
  // CHECK: 3

  return
}
