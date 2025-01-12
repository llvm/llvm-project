// RUN: mlir-opt %s -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @extract_element_0d(%a: vector<f32>) {
  %1 = vector.extractelement %a[] : vector<f32>
  // CHECK: 42
  vector.print %1: f32
  return
}

func.func @insert_element_0d(%a: f32, %b: vector<f32>) -> (vector<f32>) {
  %1 = vector.insertelement %a, %b[] : vector<f32>
  return %1: vector<f32>
}

func.func @print_vector_0d(%a: vector<f32>) {
  // CHECK: ( 42 )
  vector.print %a: vector<f32>
  return
}

func.func @splat_0d(%a: f32) {
  %1 = vector.splat %a : vector<f32>
  // CHECK: ( 42 )
  vector.print %1: vector<f32>
  return
}

func.func @broadcast_0d(%a: f32) {
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

func.func @bitcast_0d() {
  %0 = arith.constant 42 : i32
  %1 = arith.constant dense<0> : vector<i32>
  %2 = vector.insertelement %0, %1[] : vector<i32>
  %3 = vector.bitcast %2 : vector<i32> to vector<f32>
  %4 = vector.extractelement %3[] : vector<f32>
  %5 = arith.bitcast %4 : f32 to i32
  // CHECK: 42
  vector.print %5: i32
  return
}

func.func @constant_mask_0d() {
  %1 = vector.constant_mask [0] : vector<i1>
  // CHECK: ( 0 )
  vector.print %1: vector<i1>
  %2 = vector.constant_mask [1] : vector<i1>
  // CHECK: ( 1 )
  vector.print %2: vector<i1>
  return
}

func.func @arith_cmpi_0d(%smaller : vector<i32>, %bigger : vector<i32>) {
  %0 = arith.cmpi ult, %smaller, %bigger : vector<i32>
  // CHECK: ( 1 )
  vector.print %0: vector<i1>

  %1 = arith.cmpi ugt, %smaller, %bigger : vector<i32>
  // CHECK: ( 0 )
  vector.print %1: vector<i1>

  %2 = arith.cmpi eq, %smaller, %bigger : vector<i32>
  // CHECK: ( 0 )
  vector.print %2: vector<i1>

  return
}

func.func @create_mask_0d(%zero : index, %one : index) {
  %zero_mask = vector.create_mask %zero : vector<i1>
  // CHECK: ( 0 )
  vector.print %zero_mask : vector<i1>

  %one_mask = vector.create_mask %one : vector<i1>
  // CHECK: ( 1 )
  vector.print %one_mask : vector<i1>

  return
}

func.func @reduce_add(%arg0: vector<f32>) {
  %0 = vector.reduction <add>, %arg0 : vector<f32> into f32
  vector.print %0 : f32
  // CHECK: 5
  return
}

func.func @fma_0d(%four: vector<f32>) {
  %0 = vector.fma %four, %four, %four : vector<f32>
  // 4 * 4 + 4 = 20
  // CHECK: ( 20 )
  vector.print %0: vector<f32>
  return
}

func.func @transpose_0d(%arg: vector<i32>) {
  %1 = vector.transpose %arg, [] : vector<i32> to vector<i32>
  // CHECK: ( 42 )
  vector.print %1: vector<i32>
  return
}

func.func @shuffle_0d(%v0: vector<i32>, %v1: vector<i32>) {
  %1 = vector.shuffle %v0, %v1 [0, 1, 0] : vector<i32>, vector<i32>
  // CHECK: ( 42, 43, 42 )
  vector.print %1: vector<3xi32>
  return
}

func.func @entry() {
  %0 = arith.constant 42.0 : f32
  %1 = arith.constant dense<0.0> : vector<f32>
  %2 = call  @insert_element_0d(%0, %1) : (f32, vector<f32>) -> (vector<f32>)
  call  @extract_element_0d(%2) : (vector<f32>) -> ()

  %3 = arith.constant dense<42.0> : vector<f32>
  call  @print_vector_0d(%3) : (vector<f32>) -> ()

  %4 = arith.constant 42.0 : f32

  // Warning: these must be called in their textual order of definition in the
  // file to not mess up FileCheck.
  call  @splat_0d(%4) : (f32) -> ()
  call  @broadcast_0d(%4) : (f32) -> ()
  call  @bitcast_0d() : () -> ()
  call  @constant_mask_0d() : () -> ()

  %smaller = arith.constant dense<42> : vector<i32>
  %bigger = arith.constant dense<4242> : vector<i32>
  call  @arith_cmpi_0d(%smaller, %bigger) : (vector<i32>, vector<i32>) -> ()

  %zero_idx = arith.constant 0 : index
  %one_idx = arith.constant 1 : index
  call  @create_mask_0d(%zero_idx, %one_idx) : (index, index) -> ()

  %red_array = arith.constant dense<5.0> : vector<f32>
  call  @reduce_add(%red_array) : (vector<f32>) -> ()

  %5 = arith.constant dense<4.0> : vector<f32>
  call  @fma_0d(%5) : (vector<f32>) -> ()
  %6 = arith.constant dense<42> : vector<i32>
  %7 = arith.constant dense<43> : vector<i32>
  call @transpose_0d(%6) : (vector<i32>) -> ()
  call @shuffle_0d(%6, %7) : (vector<i32>, vector<i32>) -> ()

  return
}
