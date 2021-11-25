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

func @insert_element_0d(%a: f32, %b: vector<f32>) -> (vector<f32>) {
  %1 = vector.insertelement %a, %b[] : vector<f32>
  return %1: vector<f32>
}

func @print_vector_0d(%a: vector<f32>) {
  // CHECK: ( 42 )
  vector.print %a: vector<f32>
  return
}

func @entry() {
  %0 = arith.constant 42.0 : f32
  %1 = arith.constant dense<0.0> : vector<f32>
  %2 = call  @insert_element_0d(%0, %1) : (f32, vector<f32>) -> (vector<f32>)
  call  @extract_element_0d(%2) : (vector<f32>) -> ()

  %3 = arith.constant dense<42.0> : vector<f32>
  call  @print_vector_0d(%3) : (vector<f32>) -> ()

  return
}
