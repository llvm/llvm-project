// RUN: mlir-opt -convert-func-to-emitc %s -split-input-file -verify-diagnostics

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @unsuppoted_emitc_type(%arg0: i4) -> i4 {
  return %arg0 : i4
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_rank0_alloc() -> memref<i32> {
  %alloc = memref.alloc() : memref<i32>
  return %alloc : memref<i32>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_rank0_arg(%arg0: memref<i32>) -> memref<i32> {
  return %arg0 : memref<i32>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_rank1_alloc() -> memref<1xi32> {
  %alloc = memref.alloc() : memref<1xi32>
  return %alloc : memref<1xi32>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_rank1_arg(%arg0: memref<1xi32>) -> memref<1xi32> {
  return %arg0 : memref<1xi32>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_rank2_arg(%arg0: memref<1x1xi32>) -> memref<1x1xi32> {
  return %arg0 : memref<1x1xi32>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_rank1_two_elements(%arg0: memref<2xi64>) -> memref<2xi64> {
  return %arg0 : memref<2xi64>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_multiple_values(%arg0: memref<1xi32>) -> (memref<1xi32>, i32) {
  %1 = arith.constant 7 : i32
  return %arg0, %1 : memref<1xi32>, i32
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_dynamic_shape(%arg0: memref<?xi32>) -> memref<?xi32> {
  return %arg0 : memref<?xi32>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_non_identity_layout(%arg0: memref<1x1xi32, strided<[2, 1], offset: 0>>)
    -> memref<1x1xi32, strided<[2, 1], offset: 0>> {
  return %arg0 : memref<1x1xi32, strided<[2, 1], offset: 0>>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @return_unranked(%arg0: memref<*xi32>) -> memref<*xi32> {
  return %arg0 : memref<*xi32>
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @public_function(%arg0: memref<1xi64>) -> memref<1xi64> {
  return %arg0 : memref<1xi64>
}

// -----

func.func private @callee(%arg0: i64) -> i64 {
  return %arg0 : i64
}

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @caller(%arg0: memref<1xi64>, %arg1: i64) -> memref<1xi64> {
  %0 = call @callee(%arg1) : (i64) -> i64
  return %arg0 : memref<1xi64>
}
