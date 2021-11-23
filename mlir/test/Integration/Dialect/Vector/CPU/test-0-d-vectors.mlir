// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @extract_element_0d(%a: vector<f32>) {
  %1 = vector.extractelement %a[] : vector<f32>
  // CHECK: 42
  vector.print %1: f32
  return
}

func @entry() {
  %1 = arith.constant dense<42.0> : vector<f32>
  call  @extract_element_0d(%1) : (vector<f32>) -> ()
  return
}
