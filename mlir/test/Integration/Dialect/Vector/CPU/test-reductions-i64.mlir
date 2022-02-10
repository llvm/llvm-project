// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  // Construct test vector.
  %i1 = arith.constant 1: i64
  %i2 = arith.constant 2: i64
  %i3 = arith.constant 3: i64
  %i4 = arith.constant 4: i64
  %i5 = arith.constant 5: i64
  %i6 = arith.constant -1: i64
  %i7 = arith.constant -2: i64
  %i8 = arith.constant -4: i64
  %i9 = arith.constant -80: i64
  %i10 = arith.constant -16: i64
  %v0 = vector.broadcast %i1 : i64 to vector<10xi64>
  %v1 = vector.insert %i2, %v0[1] : i64 into vector<10xi64>
  %v2 = vector.insert %i3, %v1[2] : i64 into vector<10xi64>
  %v3 = vector.insert %i4, %v2[3] : i64 into vector<10xi64>
  %v4 = vector.insert %i5, %v3[4] : i64 into vector<10xi64>
  %v5 = vector.insert %i6, %v4[5] : i64 into vector<10xi64>
  %v6 = vector.insert %i7, %v5[6] : i64 into vector<10xi64>
  %v7 = vector.insert %i8, %v6[7] : i64 into vector<10xi64>
  %v8 = vector.insert %i9, %v7[8] : i64 into vector<10xi64>
  %v9 = vector.insert %i10, %v8[9] : i64 into vector<10xi64>
  vector.print %v9 : vector<10xi64>
  //
  // test vector:
  //
  // CHECK: ( 1, 2, 3, 4, 5, -1, -2, -4, -80, -16 )

  // Various vector reductions. Not full functional unit tests, but
  // a simple integration test to see if the code runs end-to-end.
  %0 = vector.reduction <add>, %v9 : vector<10xi64> into i64
  vector.print %0 : i64
  // CHECK: -88
  %1 = vector.reduction <mul>, %v9 : vector<10xi64> into i64
  vector.print %1 : i64
  // CHECK: -1228800
  %2 = vector.reduction <minsi>, %v9 : vector<10xi64> into i64
  vector.print %2 : i64
  // CHECK: -80
  %3 = vector.reduction <maxsi>, %v9 : vector<10xi64> into i64
  vector.print %3 : i64
  // CHECK: 5
  %4 = vector.reduction <and>, %v9 : vector<10xi64> into i64
  vector.print %4 : i64
  // CHECK: 0
  %5 = vector.reduction <or>, %v9 : vector<10xi64> into i64
  vector.print %5 : i64
  // CHECK: -1
  %6 = vector.reduction <xor>, %v9 : vector<10xi64> into i64
  vector.print %6 : i64
  // CHECK: -68

  return
}
