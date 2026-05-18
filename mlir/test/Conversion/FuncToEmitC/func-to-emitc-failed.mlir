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

// -----

// A symbol with the auto-generated struct name already exists but is not an
// emitc.class (here it is an emitc.func).
emitc.func @return_i32_i32() { emitc.return }
// expected-error@+2 {{symbol 'return_i32_i32' exists but is not an emitc.class}}
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @symbol_not_a_class(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}

// -----

// The existing emitc.class is not a struct (class_type != struct).
emitc.class @return_i32_i32 {
  emitc.field @field0 : i32
  emitc.field @field1 : i32
}

// expected-error@+2 {{existing class 'return_i32_i32' is not a struct}}
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @class_not_a_struct(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}

// -----

// The existing emitc.class has a method, so it cannot be used as a plain
// struct.
emitc.class struct @return_i32_i32 {
  emitc.func @method() { emitc.return }
}

// expected-error@+2 {{existing class 'return_i32_i32' has methods; expected a plain struct}}
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @class_has_methods(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}

// -----

// The existing emitc.class has fewer fields than the return types require.
emitc.class struct @return_i32_i32 {
  emitc.field @field0 : i32
}

// expected-error@+2 {{existing class 'return_i32_i32' has wrong number of fields}}
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @class_wrong_field_count(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}

// -----

// The existing emitc.class has fields with unexpected names.
emitc.class struct @return_i32_i32 {
  emitc.field @a : i32
  emitc.field @b : i32
}

// expected-error@+2 {{existing class 'return_i32_i32': unexpected field name at index 0}}
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @class_wrong_field_names(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}

// -----

// The existing emitc.class has fields with the wrong types.
emitc.class struct @return_i32_i32 {
  emitc.field @field0 : i64
  emitc.field @field1 : i32
}

// expected-error@+2 {{existing class 'return_i32_i32': wrong type for field 0}}
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @class_wrong_field_types(%arg0: i32) -> (i32, i32) {
  return %arg0, %arg0 : i32, i32
}

// -----

// Multi-result function where one result is an array type.
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func private @multi_result_with_array() -> (i32, !emitc.array<10xi32>)
