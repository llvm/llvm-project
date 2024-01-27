// RUN: mlir-opt %s -inline='default-pipeline=''' | FileCheck %s
// RUN: mlir-opt %s --mlir-disable-threading -inline='default-pipeline=''' | FileCheck %s
// RUN: mlir-opt %s -inline='default-pipeline=''' -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s --check-prefix INLINE-LOC
// RUN: mlir-opt %s -inline | FileCheck %s --check-prefix INLINE_SIMPLIFY
// RUN: mlir-opt %s -inline='op-pipelines=func.func(canonicalize,cse)' | FileCheck %s --check-prefix INLINE_SIMPLIFY

// Inline a function that takes an argument.
func.func @func_with_arg(%c : i32) -> i32 {
  %b = arith.addi %c, %c : i32
  return %b : i32
}

// CHECK-LABEL: func @inline_with_arg
func.func @inline_with_arg(%arg0 : i32) -> i32 {
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: return

  %0 = call @func_with_arg(%arg0) : (i32) -> i32
  return %0 : i32
}

// Inline a function that has multiple return operations.
func.func @func_with_multi_return(%a : i1) -> (i32) {
  cf.cond_br %a, ^bb1, ^bb2

^bb1:
  %const_0 = arith.constant 0 : i32
  return %const_0 : i32

^bb2:
  %const_55 = arith.constant 55 : i32
  return %const_55 : i32
}

// CHECK-LABEL: func @inline_with_multi_return() -> i32
func.func @inline_with_multi_return() -> i32 {
// CHECK-NEXT:    [[VAL_7:%.*]] = arith.constant false
// CHECK-NEXT:    cf.cond_br [[VAL_7]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK-NEXT:    [[VAL_8:%.*]] = arith.constant 0 : i32
// CHECK-NEXT:    cf.br ^bb3([[VAL_8]] : i32)
// CHECK:       ^bb2:
// CHECK-NEXT:    [[VAL_9:%.*]] = arith.constant 55 : i32
// CHECK-NEXT:    cf.br ^bb3([[VAL_9]] : i32)
// CHECK:       ^bb3([[VAL_10:%.*]]: i32):
// CHECK-NEXT:    return [[VAL_10]] : i32

  %false = arith.constant false
  %x = call @func_with_multi_return(%false) : (i1) -> i32
  return %x : i32
}

// Check that location information is updated for inlined instructions.
func.func @func_with_locations(%c : i32) -> i32 {
  %b = arith.addi %c, %c : i32 loc("mysource.cc":10:8)
  return %b : i32 loc("mysource.cc":11:2)
}

// INLINE-LOC-LABEL: func @inline_with_locations
func.func @inline_with_locations(%arg0 : i32) -> i32 {
  // INLINE-LOC-NEXT: arith.addi %{{.*}}, %{{.*}} : i32 loc(callsite("mysource.cc":10:8 at "mysource.cc":55:14))
  // INLINE-LOC-NEXT: return

  %0 = call @func_with_locations(%arg0) : (i32) -> i32 loc("mysource.cc":55:14)
  return %0 : i32
}


// Check that external function declarations are not inlined.
func.func private @func_external()

// CHECK-LABEL: func @no_inline_external
func.func @no_inline_external() {
  // CHECK-NEXT: call @func_external()
  call @func_external() : () -> ()
  return
}

// Check that multiple levels of calls will be inlined.
func.func @multilevel_func_a() {
  return
}
func.func @multilevel_func_b() {
  call @multilevel_func_a() : () -> ()
  return
}

// CHECK-LABEL: func @inline_multilevel
func.func @inline_multilevel() {
  // CHECK-NOT: call
  %fn = "test.functional_region_op"() ({
    call @multilevel_func_b() : () -> ()
    "test.return"() : () -> ()
  }) : () -> (() -> ())

  call_indirect %fn() : () -> ()
  return
}

// Check that recursive calls are not inlined.
// CHECK-LABEL: func @no_inline_recursive
func.func @no_inline_recursive() {
  // CHECK: test.functional_region_op
  // CHECK-NOT: test.functional_region_op
  %fn = "test.functional_region_op"() ({
    call @no_inline_recursive() : () -> ()
    "test.return"() : () -> ()
  }) : () -> (() -> ())
  return
}

// Check that we can convert types for inputs and results as necessary.
func.func @convert_callee_fn(%arg : i32) -> i32 {
  return %arg : i32
}
func.func @convert_callee_fn_multi_arg(%a : i32, %b : i32) -> () {
  return
}
func.func @convert_callee_fn_multi_res() -> (i32, i32) {
  %res = arith.constant 0 : i32
  return %res, %res : i32, i32
}

// CHECK-LABEL: func @inline_convert_call
func.func @inline_convert_call() -> i16 {
  // CHECK: %[[INPUT:.*]] = arith.constant
  %test_input = arith.constant 0 : i16

  // CHECK: %[[CAST_INPUT:.*]] = "test.cast"(%[[INPUT]]) : (i16) -> i32
  // CHECK: %[[CAST_RESULT:.*]] = "test.cast"(%[[CAST_INPUT]]) : (i32) -> i16
  // CHECK-NEXT: return %[[CAST_RESULT]]
  %res = "test.conversion_call_op"(%test_input) { callee=@convert_callee_fn } : (i16) -> (i16)
  return %res : i16
}

func.func @convert_callee_fn_multiblock() -> i32 {
  cf.br ^bb0
^bb0:
  %0 = arith.constant 0 : i32
  return %0 : i32
}

// CHECK-LABEL: func @inline_convert_result_multiblock
func.func @inline_convert_result_multiblock() -> i16 {
// CHECK:   cf.br ^bb1 {inlined_conversion}
// CHECK: ^bb1:
// CHECK:   %[[C:.+]] = arith.constant {inlined_conversion} 0 : i32
// CHECK:   cf.br ^bb2(%[[C]] : i32)
// CHECK: ^bb2(%[[BBARG:.+]]: i32):
// CHECK:   %[[CAST_RESULT:.+]] = "test.cast"(%[[BBARG]]) : (i32) -> i16
// CHECK:   return %[[CAST_RESULT]] : i16

  %res = "test.conversion_call_op"() { callee=@convert_callee_fn_multiblock } : () -> (i16)
  return %res : i16
}

// CHECK-LABEL: func @no_inline_convert_call
func.func @no_inline_convert_call() {
  // CHECK: "test.conversion_call_op"
  %test_input_i16 = arith.constant 0 : i16
  %test_input_i64 = arith.constant 0 : i64
  "test.conversion_call_op"(%test_input_i16, %test_input_i64) { callee=@convert_callee_fn_multi_arg } : (i16, i64) -> ()

  // CHECK: "test.conversion_call_op"
  %res_2:2 = "test.conversion_call_op"() { callee=@convert_callee_fn_multi_res } : () -> (i16, i64)
  return
}

// Check that we properly simplify when inlining.
func.func @simplify_return_constant() -> i32 {
  %res = arith.constant 0 : i32
  return %res : i32
}

func.func @simplify_return_reference() -> (() -> i32) {
  %res = constant @simplify_return_constant : () -> i32
  return %res : () -> i32
}

// INLINE_SIMPLIFY-LABEL: func @inline_simplify
func.func @inline_simplify() -> i32 {
  // INLINE_SIMPLIFY-NEXT: %[[CST:.*]] = arith.constant 0 : i32
  // INLINE_SIMPLIFY-NEXT: return %[[CST]]
  %fn = call @simplify_return_reference() : () -> (() -> i32)
  %res = call_indirect %fn() : () -> i32
  return %res : i32
}

// CHECK-LABEL: func @no_inline_invalid_call
func.func @no_inline_invalid_call() -> i32 {
  %res = "test.conversion_call_op"() { callee=@convert_callee_fn_multiblock, noinline } : () -> (i32)
  return %res : i32
}

func.func @gpu_alloc() -> memref<1024xf32> {
  %m = gpu.alloc [] () : memref<1024xf32>
  return %m : memref<1024xf32>
}

// CHECK-LABEL: func @inline_gpu_ops
func.func @inline_gpu_ops() -> memref<1024xf32> {
  // CHECK-NEXT: gpu.alloc
  %m = call @gpu_alloc() : () -> memref<1024xf32>
  return %m : memref<1024xf32>
}

// Test block arguments location propagation.
// Use two call-sites to force cloning.
func.func @func_with_block_args_location(%arg0 : i32) {
  cf.br ^bb1(%arg0 : i32)
^bb1(%x : i32 loc("foo")):
  "test.foo" (%x) : (i32) -> () loc("bar")
  return
}

// INLINE-LOC-LABEL: func @func_with_block_args_location_callee1
// INLINE-LOC: cf.br
// INLINE-LOC: ^bb{{[0-9]+}}(%{{.*}}: i32 loc("foo")
func.func @func_with_block_args_location_callee1(%arg0 : i32) {
  call @func_with_block_args_location(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL: func @func_with_block_args_location_callee2
func.func @func_with_block_args_location_callee2(%arg0 : i32) {
  call @func_with_block_args_location(%arg0) : (i32) -> ()
  return
}

func.func @func_with_multiple_blocks(%arg0 : i32) {
  cf.br ^bb1(%arg0 : i32)
^bb1(%x : i32):
  "test.foo" (%x) : (i32) -> () loc("bar")
  return
}

// CHECK-LABEL: func @func_with_multiple_blocks_callee1
func.func @func_with_multiple_blocks_callee1(%arg0 : i32) {
  "test.dummy_op"() ({
    // Call cannot be inlined because "test.dummy" may not support unstructured
    // control flow in its body.
    // CHECK: call @func_with_multiple_blocks
    call @func_with_multiple_blocks(%arg0) : (i32) -> ()
    "test.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @func_with_multiple_blocks_callee2
func.func @func_with_multiple_blocks_callee2(%arg0 : i32, %c : i1) {
  %0 = scf.while (%arg1 = %arg0) : (i32) -> (i32) {
    // Call cannot be inlined because scf.while does not support unstructured
    // control flow in its body.
    // CHECK: call @func_with_multiple_blocks
    func.call @func_with_multiple_blocks(%arg0) : (i32) -> ()
    scf.condition(%c) %arg1 : i32
  } do {
  ^bb0(%arg1: i32):
    scf.yield %arg1 : i32
  }
  return
}

// Check that we can handle argument and result attributes.
test.conversion_func_op @handle_attr_callee_fn_multi_arg(%arg0 : i16, %arg1 : i16 {"test.handle_argument"}) -> (i16 {"test.handle_result"}, i16) {
  %0 = arith.addi %arg0, %arg1 : i16
  %1 = arith.subi %arg0, %arg1 : i16
  "test.return"(%0, %1) : (i16, i16) -> ()
}
test.conversion_func_op @handle_attr_callee_fn(%arg0 : i32 {"test.handle_argument"}) -> (i32 {"test.handle_result"}) {
  "test.return"(%arg0) : (i32) -> ()
}

// CHECK-LABEL: func @inline_handle_attr_call
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]
func.func @inline_handle_attr_call(%arg0 : i16, %arg1 : i16) -> (i16, i16) {

  // CHECK: %[[CHANGE_INPUT:.*]] = "test.type_changer"(%[[ARG1]]) : (i16) -> i16
  // CHECK: %[[SUM:.*]] = arith.addi %[[ARG0]], %[[CHANGE_INPUT]]
  // CHECK: %[[DIFF:.*]] = arith.subi %[[ARG0]], %[[CHANGE_INPUT]]
  // CHECK: %[[CHANGE_RESULT:.*]] = "test.type_changer"(%[[SUM]]) : (i16) -> i16
  // CHECK-NEXT: return %[[CHANGE_RESULT]], %[[DIFF]]
  %res0, %res1 = "test.conversion_call_op"(%arg0, %arg1) { callee=@handle_attr_callee_fn_multi_arg } : (i16, i16) -> (i16, i16)
  return %res0, %res1 : i16, i16
}

// CHECK-LABEL: func @inline_convert_and_handle_attr_call
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]
func.func @inline_convert_and_handle_attr_call(%arg0 : i16) -> (i16) {

  // CHECK: %[[CAST_INPUT:.*]] = "test.cast"(%[[ARG0]]) : (i16) -> i32
  // CHECK: %[[CHANGE_INPUT:.*]] = "test.type_changer"(%[[CAST_INPUT]]) : (i32) -> i32
  // CHECK: %[[CHANGE_RESULT:.*]] = "test.type_changer"(%[[CHANGE_INPUT]]) : (i32) -> i32
  // CHECK: %[[CAST_RESULT:.*]] = "test.cast"(%[[CHANGE_RESULT]]) : (i32) -> i16
  // CHECK: return %[[CAST_RESULT]]
  %res = "test.conversion_call_op"(%arg0) { callee=@handle_attr_callee_fn } : (i16) -> (i16)
  return %res : i16
}

// Check a function with complex ops is inlined.
func.func @double_square_complex(%cplx: complex<f32>) -> complex<f32> {
  %double = complex.add %cplx, %cplx : complex<f32>
  %square = complex.mul %double, %double : complex<f32>
  return %square : complex<f32>
}

// CHECK-LABEL: func @inline_with_complex_ops
func.func @inline_with_complex_ops() -> complex<f32> {
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %c = complex.create %c1, %c2 : complex<f32>

  // CHECK: complex.add
  // CHECK: complex.mul
  // CHECK-NOT: call
  %r = call @double_square_complex(%c) : (complex<f32>) -> (complex<f32>)
  return %r : complex<f32>
}
