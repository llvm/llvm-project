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

func @splat_0d(%a: f32) {
  %1 = splat %a : vector<f32>
  // CHECK: ( 42 )
  vector.print %1: vector<f32>
  return
}

func @broadcast_0d(%a: f32) {
  %1 = vector.broadcast %a : f32 to vector<f32>
  // CHECK: ( 42 )
  vector.print %1: vector<f32>

  %2 = vector.broadcast %1 : vector<f32> to vector<f32>
  // CHECK: ( 42 )
  vector.print %2: vector<f32>

  %3 = vector.broadcast %1 : vector<f32> to vector<1xf32>
  // CHECK: ( 42 )
  vector.print %3: vector<1xf32>

  %4 = vector.broadcast %1 : vector<f32> to vector<2xf32>
  // CHECK: ( 42, 42 )
  vector.print %4: vector<2xf32>

  %5 = vector.broadcast %1 : vector<f32> to vector<2x1xf32>
  // CHECK: ( ( 42 ), ( 42 ) )
  vector.print %5: vector<2x1xf32>

  %6 = vector.broadcast %1 : vector<f32> to vector<2x3xf32>
  // CHECK: ( ( 42, 42, 42 ), ( 42, 42, 42 ) )
  vector.print %6: vector<2x3xf32>
  return
}

func @entry() {
  %0 = arith.constant 42.0 : f32
  %1 = arith.constant dense<0.0> : vector<f32>
  %2 = call  @insert_element_0d(%0, %1) : (f32, vector<f32>) -> (vector<f32>)
  call  @extract_element_0d(%2) : (vector<f32>) -> ()

  %3 = arith.constant dense<42.0> : vector<f32>
  call  @print_vector_0d(%3) : (vector<f32>) -> ()

  %4 = arith.constant 42.0 : f32
  call  @splat_0d(%4) : (f32) -> ()
  call  @broadcast_0d(%4) : (f32) -> ()

  return
}
