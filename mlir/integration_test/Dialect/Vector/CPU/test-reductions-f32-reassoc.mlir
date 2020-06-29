// RUN: mlir-opt %s -convert-scf-to-std \
// RUN:             -convert-vector-to-llvm='reassociate-fp-reductions' \
// RUN:             -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  // Construct test vector, numerically very stable.
  %f1 = constant 1.0: f32
  %f2 = constant 2.0: f32
  %f3 = constant 3.0: f32
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
  %0 = vector.reduction "add", %v2 : vector<64xf32> into f32
  vector.print %0 : f32
  // CHECK: 67
  %1 = vector.reduction "mul", %v2 : vector<64xf32> into f32
  vector.print %1 : f32
  // CHECK: 6
  %2 = vector.reduction "min", %v2 : vector<64xf32> into f32
  vector.print %2 : f32
  // CHECK: 1
  %3 = vector.reduction "max", %v2 : vector<64xf32> into f32
  vector.print %3 : f32
  // CHECK: 3

  return
}
