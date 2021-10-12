// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @entry() {
  %v0 = arith.constant dense<[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]> : vector<16xi4>
  %v = vector.bitcast %v0 : vector<16xi4> to vector<16xsi4>
  vector.print %v : vector<16xsi4>
  //
  // Test vector:
  //
  // CHECK: ( -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7 )

  %0 = vector.reduction "add", %v : vector<16xsi4> into si4
  vector.print %0 : si4
  // CHECK: -8

  %1 = vector.reduction "mul", %v : vector<16xsi4> into si4
  vector.print %1 : si4
  // CHECK: 0

  %2 = vector.reduction "minsi", %v : vector<16xsi4> into si4
  vector.print %2 : si4
  // CHECK: -8

  %3 = vector.reduction "maxsi", %v : vector<16xsi4> into si4
  vector.print %3 : si4
  // CHECK: 7

  %4 = vector.reduction "and", %v : vector<16xsi4> into si4
  vector.print %4 : si4
  // CHECK: 0

  %5 = vector.reduction "or", %v : vector<16xsi4> into si4
  vector.print %5 : si4
  // CHECK: -1

  %6 = vector.reduction "xor", %v : vector<16xsi4> into si4
  vector.print %6 : si4
  // CHECK: 0

  return
}
