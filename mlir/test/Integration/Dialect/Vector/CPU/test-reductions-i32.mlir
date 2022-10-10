// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @entry() {
  // Construct test vector.
  %i1 = arith.constant 1: i32
  %i2 = arith.constant 2: i32
  %i3 = arith.constant 3: i32
  %i4 = arith.constant 4: i32
  %i5 = arith.constant 5: i32
  %i6 = arith.constant -1: i32
  %i7 = arith.constant -2: i32
  %i8 = arith.constant -4: i32
  %i9 = arith.constant -80: i32
  %i10 = arith.constant -16: i32
  %v0 = vector.broadcast %i1 : i32 to vector<10xi32>
  %v1 = vector.insert %i2, %v0[1] : i32 into vector<10xi32>
  %v2 = vector.insert %i3, %v1[2] : i32 into vector<10xi32>
  %v3 = vector.insert %i4, %v2[3] : i32 into vector<10xi32>
  %v4 = vector.insert %i5, %v3[4] : i32 into vector<10xi32>
  %v5 = vector.insert %i6, %v4[5] : i32 into vector<10xi32>
  %v6 = vector.insert %i7, %v5[6] : i32 into vector<10xi32>
  %v7 = vector.insert %i8, %v6[7] : i32 into vector<10xi32>
  %v8 = vector.insert %i9, %v7[8] : i32 into vector<10xi32>
  %v9 = vector.insert %i10, %v8[9] : i32 into vector<10xi32>
  vector.print %v9 : vector<10xi32>
  //
  // test vector:
  //
  // CHECK: ( 1, 2, 3, 4, 5, -1, -2, -4, -80, -16 )

  // Various vector reductions. Not full functional unit tests, but
  // a simple integration test to see if the code runs end-to-end.
  %0 = vector.reduction <add>, %v9 : vector<10xi32> into i32
  vector.print %0 : i32
  // CHECK: -88
  %1 = vector.reduction <mul>, %v9 : vector<10xi32> into i32
  vector.print %1 : i32
  // CHECK: -1228800
  %2 = vector.reduction <minsi>, %v9 : vector<10xi32> into i32
  vector.print %2 : i32
  // CHECK: -80
  %3 = vector.reduction <maxsi>, %v9 : vector<10xi32> into i32
  vector.print %3 : i32
  // CHECK: 5
  %4 = vector.reduction <and>, %v9 : vector<10xi32> into i32
  vector.print %4 : i32
  // CHECK: 0
  %5 = vector.reduction <or>, %v9 : vector<10xi32> into i32
  vector.print %5 : i32
  // CHECK: -1
  %6 = vector.reduction <xor>, %v9 : vector<10xi32> into i32
  vector.print %6 : i32
  // CHECK: -68

  return
}
