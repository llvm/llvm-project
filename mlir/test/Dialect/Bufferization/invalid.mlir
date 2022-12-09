// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @alloc_tensor_missing_dims(%arg0: index)
{
  // expected-error @+1 {{expected 2 dynamic sizes}}
  %0 = bufferization.alloc_tensor(%arg0) : tensor<4x?x?x5xf32>
  return
}

// -----

// expected-note @+1 {{prior use here}}
func.func @alloc_tensor_type_mismatch(%t: tensor<?xf32>) {
  // expected-error @+1{{expects different type than prior uses: 'tensor<5xf32>' vs 'tensor<?xf32>'}}
  %0 = bufferization.alloc_tensor() copy(%t) : tensor<5xf32>
  return
}

// -----

func.func @alloc_tensor_copy_and_dims(%t: tensor<?xf32>, %sz: index) {
  // expected-error @+1{{dynamic sizes not needed when copying a tensor}}
  %0 = bufferization.alloc_tensor(%sz) copy(%t) : tensor<?xf32>
  return
}

// -----

func.func @alloc_tensor_invalid_escape_attr(%sz: index) {
  // expected-error @+1{{'bufferization.escape' is expected to be a bool array attribute}}
  %0 = bufferization.alloc_tensor(%sz) {bufferization.escape = 5} : tensor<?xf32>
  return
}

// -----

func.func @alloc_tensor_invalid_escape_attr_size(%sz: index) {
  // expected-error @+1{{'bufferization.escape' has wrong number of elements, expected 1, got 2}}
  %0 = bufferization.alloc_tensor(%sz) {bufferization.escape = [true, false]} : tensor<?xf32>
  return
}

// -----

func.func @escape_attr_non_allocating(%t0: tensor<?xf32>) {
  // expected-error @+1{{'bufferization.escape' only valid for allocation results}}
  %0 = tensor.extract_slice %t0[0][5][1] {bufferization.escape = [true]} : tensor<?xf32> to tensor<5xf32>
  return
}

// -----

func.func @escape_attr_non_bufferizable(%m0: memref<?xf32>) {
  // expected-error @+1{{'bufferization.escape' only valid on bufferizable ops}}
  %0 = memref.cast %m0 {bufferization.escape = [true]} : memref<?xf32> to memref<10xf32>
  return
}

// -----

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

func.func @sparse_alloc_direct_return() -> tensor<20x40xf32, #DCSR> {
  // expected-error @+1{{sparse tensor allocation should not escape function}}
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  return %0 : tensor<20x40xf32, #DCSR>
}

// -----

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

func.func private @foo(tensor<20x40xf32, #DCSR>) -> ()

func.func @sparse_alloc_call() {
  // expected-error @+1{{sparse tensor allocation should not escape function}}
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  call @foo(%0) : (tensor<20x40xf32, #DCSR>) -> ()
  return
}
