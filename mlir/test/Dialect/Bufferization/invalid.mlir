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

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

func.func @sparse_alloc_direct_return() -> tensor<20x40xf32, #DCSR> {
  // expected-error @+1{{sparse tensor allocation should not escape function}}
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  return %0 : tensor<20x40xf32, #DCSR>
}

// -----

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

func.func private @foo(tensor<20x40xf32, #DCSR>) -> ()

func.func @sparse_alloc_call() {
  // expected-error @+1{{sparse tensor allocation should not escape function}}
  %0 = bufferization.alloc_tensor() : tensor<20x40xf32, #DCSR>
  call @foo(%0) : (tensor<20x40xf32, #DCSR>) -> ()
  return
}

// -----

// expected-error @+1{{invalid value for 'bufferization.access'}}
func.func private @invalid_buffer_access_type(tensor<*xf32> {bufferization.access = "foo"})

// -----

// expected-error @+1{{'bufferization.writable' is invalid on external functions}}
func.func private @invalid_writable_attribute(tensor<*xf32> {bufferization.writable = false})

// -----

func.func @invalid_writable_on_op() {
  // expected-error @+1{{attribute '"bufferization.writable"' not supported as an op attribute by the bufferization dialect}}
  arith.constant {bufferization.writable = true} 0  : index
}

// -----

func.func @invalid_materialize_in_destination(%arg0: tensor<?xf32>, %arg1: tensor<5xf32>) {
  // expected-error @below{{failed to verify that all of {source, dest} have same shape}}
  bufferization.materialize_in_destination %arg0 in %arg1 : (tensor<?xf32>, tensor<5xf32>) -> tensor<5xf32>
}

// -----

func.func @invalid_materialize_in_destination_dest_type(%arg0: tensor<5xf32>, %arg1: vector<5xf32>) {
  // expected-error @below{{'dest' must be a tensor or a memref}}
  bufferization.materialize_in_destination %arg0 in %arg1 : (tensor<5xf32>, vector<5xf32>) -> ()
}

// -----

func.func @invalid_materialize_in_destination_result(%arg0: tensor<?xf32>, %arg1: memref<?xf32>) {
  // expected-error @below{{memref 'dest' implies zero results}}
  bufferization.materialize_in_destination %arg0 in restrict %arg1 : (tensor<?xf32>, memref<?xf32>) -> (tensor<?xf32>)
}

// -----

func.func @invalid_materialize_in_destination_result_missing(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // expected-error @below{{tensor 'dest' implies exactly one tensor result}}
  bufferization.materialize_in_destination %arg0 in %arg1 : (tensor<?xf32>, tensor<?xf32>) -> ()
}

// -----

func.func @invalid_materialize_in_destination_restrict(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // expected-error @below{{'restrict' is valid only for memref destinations}}
  bufferization.materialize_in_destination %arg0 in restrict %arg1 : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
}

// -----

func.func @invalid_materialize_in_destination_restrict(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // expected-error @below{{'writable' must be specified if and only if the destination is of memref type}}
  bufferization.materialize_in_destination %arg0 in writable %arg1 : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
}

// -----

func.func @invalid_materialize_in_destination_result_shape(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // expected-error @below{{result and 'dest' types must match}}
  bufferization.materialize_in_destination %arg0 in %arg1 : (tensor<?xf32>, tensor<?xf32>) -> (tensor<6xf32>)
}

// -----

func.func @invalid_dealloc_memref_condition_mismatch(%arg0: memref<2xf32>, %arg1: memref<4xi32>, %arg2: i1) {
  // expected-error @below{{must have the same number of conditions as memrefs to deallocate}}
  bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<4xi32>) if (%arg2)
  return
}

// -----

func.func @invalid_dealloc_wrong_number_of_results(%arg0: memref<2xf32>, %arg1: memref<4xi32>, %arg2: i1) -> i1 {
  // expected-error @below{{operation defines 1 results but was provided 2 to bind}}
  %0:2 = bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<4xi32>) if (%arg2, %arg2) retain (%arg1 : memref<4xi32>)
  return %0#0 : i1
}

// -----

func.func @invalid_dealloc_wrong_number_of_results(%arg0: memref<2xf32>, %arg1: memref<4xi32>, %arg2: i1) -> i1 {
  // expected-error @below{{must have the same number of updated conditions (results) as retained operands}}
  %0:3 = "bufferization.dealloc"(%arg0, %arg1, %arg2, %arg2, %arg1) <{operandSegmentSizes = array<i32: 2, 2, 1>}> : (memref<2xf32>, memref<4xi32>, i1, i1, memref<4xi32>) -> (i1, i1, i1)
  return %0#0 : i1
}

// -----

func.func @invalid_manual_deallocation() {
  // expected-error @below{{op attribute 'bufferization.manual_deallocation' can be used only on ops that have an allocation and/or free side effect}}
  arith.constant {bufferization.manual_deallocation} 0  : index
}
