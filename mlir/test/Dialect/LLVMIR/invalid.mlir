// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// expected-error@+1{{llvm.noalias argument attribute of non boolean type}}
func @invalid_noalias(%arg0: i32 {llvm.noalias = 3}) {
  "llvm.return"() : () -> ()
}

// -----

// expected-error@+1{{llvm.align argument attribute of non integer type}}
func @invalid_align(%arg0: i32 {llvm.align = "foo"}) {
  "llvm.return"() : () -> ()
}

////////////////////////////////////////////////////////////////////////////////

// Check that parser errors are properly produced and do not crash the compiler.

// -----

func @icmp_non_string(%arg0 : i32, %arg1 : i16) {
  // expected-error@+1 {{invalid kind of attribute specified}}
  llvm.icmp 42 %arg0, %arg0 : i32
  return
}

// -----

func @icmp_wrong_string(%arg0 : i32, %arg1 : i16) {
  // expected-error@+1 {{'foo' is an incorrect value of the 'predicate' attribute}}
  llvm.icmp "foo" %arg0, %arg0 : i32
  return
}

// -----

func @alloca_missing_input_result_type(%size : i64) {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : () -> ()
}

// -----

func @alloca_missing_input_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : () -> (!llvm.ptr<i32>)
}

// -----

func @alloca_missing_result_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : (i64) -> ()
}

// -----

func @alloca_non_function_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : !llvm.ptr<i32>
}

// -----

func @alloca_non_integer_alignment() {
  // expected-error@+1 {{expected integer alignment}}
  llvm.alloca %size x i32 {alignment = 3.0} : !llvm.ptr<i32>
}

// -----

func @gep_missing_input_result_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{2 operands present, but expected 0}}
  llvm.getelementptr %base[%pos] : () -> ()
}

// -----

func @gep_missing_input_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{2 operands present, but expected 0}}
  llvm.getelementptr %base[%pos] : () -> (!llvm.ptr<f32>)
}

// -----

func @gep_missing_result_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{op requires one result}}
  llvm.getelementptr %base[%pos] : (!llvm.ptr<f32>, i64) -> ()
}

// -----

func @gep_non_function_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{invalid kind of type specified}}
  llvm.getelementptr %base[%pos] : !llvm.ptr<f32>
}

// -----

func @load_non_llvm_type(%foo : memref<f32>) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.load %foo : memref<f32>
}

// -----

func @load_non_ptr_type(%foo : f32) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.load %foo : f32
}

// -----

func @store_non_llvm_type(%foo : memref<f32>, %bar : f32) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.store %bar, %foo : memref<f32>
}

// -----

func @store_non_ptr_type(%foo : f32, %bar : f32) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.store %bar, %foo : f32
}

// -----

func @call_non_function_type(%callee : !llvm.func<i8 (i8)>, %arg : i8) {
  // expected-error@+1 {{expected function type}}
  llvm.call %callee(%arg) : !llvm.func<i8 (i8)>
}

// -----

func @invalid_call() {
  // expected-error@+1 {{'llvm.call' op must have either a `callee` attribute or at least an operand}}
  "llvm.call"() : () -> ()
}

// -----

func @call_non_function_type(%callee : !llvm.func<i8 (i8)>, %arg : i8) {
  // expected-error@+1 {{expected function type}}
  llvm.call %callee(%arg) : !llvm.func<i8 (i8)>
}

// -----

func @call_unknown_symbol() {
  // expected-error@+1 {{'llvm.call' op 'missing_callee' does not reference a symbol in the current scope}}
  llvm.call @missing_callee() : () -> ()
}

// -----

func private @standard_func_callee()

func @call_non_llvm() {
  // expected-error@+1 {{'llvm.call' op 'standard_func_callee' does not reference a valid LLVM function}}
  llvm.call @standard_func_callee() : () -> ()
}

// -----

func @call_non_llvm_indirect(%arg0 : tensor<*xi32>) {
  // expected-error@+1 {{'llvm.call' op operand #0 must be LLVM dialect-compatible type}}
  "llvm.call"(%arg0) : (tensor<*xi32>) -> ()
}

// -----

llvm.func @callee_func(i8) -> ()

func @callee_arg_mismatch(%arg0 : i32) {
  // expected-error@+1 {{'llvm.call' op operand type mismatch for operand 0: 'i32' != 'i8'}}
  llvm.call @callee_func(%arg0) : (i32) -> ()
}

// -----

func @indirect_callee_arg_mismatch(%arg0 : i32, %callee : !llvm.ptr<func<void(i8)>>) {
  // expected-error@+1 {{'llvm.call' op operand type mismatch for operand 0: 'i32' != 'i8'}}
  "llvm.call"(%callee, %arg0) : (!llvm.ptr<func<void(i8)>>, i32) -> ()
}

// -----

llvm.func @callee_func() -> (i8)

func @callee_return_mismatch() {
  // expected-error@+1 {{'llvm.call' op result type mismatch: 'i32' != 'i8'}}
  %res = llvm.call @callee_func() : () -> (i32)
}

// -----

func @indirect_callee_return_mismatch(%callee : !llvm.ptr<func<i8()>>) {
  // expected-error@+1 {{'llvm.call' op result type mismatch: 'i32' != 'i8'}}
  "llvm.call"(%callee) : (!llvm.ptr<func<i8()>>) -> (i32)
}

// -----

func @call_too_many_results(%callee : () -> (i32,i32)) {
  // expected-error@+1 {{expected function with 0 or 1 result}}
  llvm.call %callee() : () -> (i32, i32)
}

// -----

func @call_non_llvm_result(%callee : () -> (tensor<*xi32>)) {
  // expected-error@+1 {{expected result to have LLVM type}}
  llvm.call %callee() : () -> (tensor<*xi32>)
}

// -----

func @call_non_llvm_input(%callee : (tensor<*xi32>) -> (), %arg : tensor<*xi32>) {
  // expected-error@+1 {{expected LLVM types as inputs}}
  llvm.call %callee(%arg) : (tensor<*xi32>) -> ()
}

// -----

func @constant_wrong_type() {
  // expected-error@+1 {{only supports integer, float, string or elements attributes}}
  llvm.mlir.constant(@constant_wrong_type) : !llvm.ptr<func<void ()>>
}

// -----

func @insertvalue_non_llvm_type(%a : i32, %b : i32) {
  // expected-error@+1 {{expected LLVM IR Dialect type}}
  llvm.insertvalue %a, %b[0] : tensor<*xi32>
}

// -----

func @insertvalue_non_array_position() {
  // Note the double-type, otherwise attribute parsing consumes the trailing
  // type of the op as the (wrong) attribute type.
  // expected-error@+1 {{invalid kind of attribute specified}}
  llvm.insertvalue %a, %b 0 : i32 : !llvm.struct<(i32)>
}

// -----

func @insertvalue_non_integer_position() {
  // expected-error@+1 {{expected an array of integer literals}}
  llvm.insertvalue %a, %b[0.0] : !llvm.struct<(i32)>
}

// -----

func @insertvalue_struct_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm.struct<(i32)>
}

// -----

func @insertvalue_array_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm.array<1 x i32>
}

// -----

func @insertvalue_wrong_nesting() {
  // expected-error@+1 {{expected LLVM IR structure/array type}}
  llvm.insertvalue %a, %b[0,0] : !llvm.struct<(i32)>
}

// -----

func @extractvalue_non_llvm_type(%a : i32, %b : tensor<*xi32>) {
  // expected-error@+1 {{expected LLVM IR Dialect type}}
  llvm.extractvalue %b[0] : tensor<*xi32>
}

// -----

func @extractvalue_non_array_position() {
  // Note the double-type, otherwise attribute parsing consumes the trailing
  // type of the op as the (wrong) attribute type.
  // expected-error@+1 {{invalid kind of attribute specified}}
  llvm.extractvalue %b 0 : i32 : !llvm.struct<(i32)>
}

// -----

func @extractvalue_non_integer_position() {
  // expected-error@+1 {{expected an array of integer literals}}
  llvm.extractvalue %b[0.0] : !llvm.struct<(i32)>
}

// -----

func @extractvalue_struct_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm.struct<(i32)>
}

// -----

func @extractvalue_array_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm.array<1 x i32>
}

// -----

func @extractvalue_wrong_nesting() {
  // expected-error@+1 {{expected LLVM IR structure/array type}}
  llvm.extractvalue %b[0,0] : !llvm.struct<(i32)>
}

// -----

func @invalid_vector_type_1(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{expected LLVM dialect-compatible vector type for operand #1}}
  %0 = llvm.extractelement %arg2[%arg1 : i32] : f32
}

// -----

func @invalid_vector_type_2(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{expected LLVM dialect-compatible vector type for operand #1}}
  %0 = llvm.insertelement %arg2, %arg2[%arg1 : i32] : f32
}

// -----

func @invalid_vector_type_3(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{expected LLVM IR dialect vector type for operand #1}}
  %0 = llvm.shufflevector %arg2, %arg2 [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : f32, f32
}

// -----

func @null_non_llvm_type() {
  // expected-error@+1 {{must be LLVM pointer type, but got 'i32'}}
  llvm.mlir.null : i32
}

// -----

func @nvvm_invalid_shfl_pred_1(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32
}

// -----

func @nvvm_invalid_shfl_pred_2(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm.struct<(i32)>
}

// -----

func @nvvm_invalid_shfl_pred_3(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm.struct<(i32, i32)>
}

// -----

func @nvvm_invalid_mma_0(%a0 : f16, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{expected operands to be 4 <halfx2>s followed by either 4 <halfx2>s or 8 floats}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (f16, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func @nvvm_invalid_mma_1(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{expected result type to be a struct of either 4 <halfx2>s or 8 floats}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>
}

// -----

func @nvvm_invalid_mma_2(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{alayout and blayout attributes must be set to either "row" or "col"}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func @nvvm_invalid_mma_3(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : vector<2xf16>, %c1 : vector<2xf16>,
                         %c2 : vector<2xf16>, %c3 : vector<2xf16>) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3 {alayout="row", blayout="col"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func @nvvm_invalid_mma_4(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
}

// -----

func @nvvm_invalid_mma_5(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func @nvvm_invalid_mma_6(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{invalid kind of type specified}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func @nvvm_invalid_mma_7(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{op requires one result}}
  %0:2 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> (!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>, i32)
  llvm.return %0#0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func @atomicrmw_expected_ptr(%f32 : f32) {
  // expected-error@+1 {{operand #0 must be LLVM pointer to floating point LLVM type or LLVM integer type}}
  %0 = "llvm.atomicrmw"(%f32, %f32) {bin_op=11, ordering=1} : (f32, f32) -> f32
  llvm.return
}

// -----

func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr<f32>, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR element type for operand #0 to match type for operand #1}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %i32) {bin_op=11, ordering=1} : (!llvm.ptr<f32>, i32) -> f32
  llvm.return
}

// -----

func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr<f32>, %f32 : f32) {
  // expected-error@+1 {{expected LLVM IR result type to match type for operand #1}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %f32) {bin_op=11, ordering=1} : (!llvm.ptr<f32>, f32) -> i32
  llvm.return
}

// -----

func @atomicrmw_expected_float(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR floating point type}}
  %0 = llvm.atomicrmw fadd %i32_ptr, %i32 unordered : i32
  llvm.return
}

// -----

func @atomicrmw_unexpected_xchg_type(%i1_ptr : !llvm.ptr<i1>, %i1 : i1) {
  // expected-error@+1 {{unexpected LLVM IR type for 'xchg' bin_op}}
  %0 = llvm.atomicrmw xchg %i1_ptr, %i1 unordered : i1
  llvm.return
}

// -----

func @atomicrmw_expected_int(%f32_ptr : !llvm.ptr<f32>, %f32 : f32) {
  // expected-error@+1 {{expected LLVM IR integer type}}
  %0 = llvm.atomicrmw max %f32_ptr, %f32 unordered : f32
  llvm.return
}

// -----

func @cmpxchg_expected_ptr(%f32_ptr : !llvm.ptr<f32>, %f32 : f32) {
  // expected-error@+1 {{op operand #0 must be LLVM pointer to LLVM integer type or LLVM pointer type}}
  %0 = "llvm.cmpxchg"(%f32, %f32, %f32) {success_ordering=2,failure_ordering=2} : (f32, f32, f32) -> !llvm.struct<(f32, i1)>
  llvm.return
}

// -----

func @cmpxchg_mismatched_operands(%i64_ptr : !llvm.ptr<i64>, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR element type for operand #0 to match type for all other operands}}
  %0 = "llvm.cmpxchg"(%i64_ptr, %i32, %i32) {success_ordering=2,failure_ordering=2} : (!llvm.ptr<i64>, i32, i32) -> !llvm.struct<(i32, i1)>
  llvm.return
}

// -----

func @cmpxchg_unexpected_type(%i1_ptr : !llvm.ptr<i1>, %i1 : i1) {
  // expected-error@+1 {{unexpected LLVM IR type}}
  %0 = llvm.cmpxchg %i1_ptr, %i1, %i1 monotonic monotonic : i1
  llvm.return
}

// -----

func @cmpxchg_at_least_monotonic_success(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 unordered monotonic : i32
  llvm.return
}

// -----

func @cmpxchg_at_least_monotonic_failure(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 monotonic unordered : i32
  llvm.return
}

// -----

func @cmpxchg_failure_release(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel release : i32
  llvm.return
}

// -----

func @cmpxchg_failure_acq_rel(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel acq_rel : i32
  llvm.return
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @bad_landingpad(%arg0: !llvm.ptr<ptr<i8>>) attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(3 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.invoke @foo(%1) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1:  // pred: ^bb0
  llvm.return %1 : i32
^bb2:  // pred: ^bb0
  // expected-error@+1 {{clause #0 is not a known constant - null, addressof, bitcast}}
  %3 = llvm.landingpad cleanup (catch %1 : i32) (catch %arg0 : !llvm.ptr<ptr<i8>>) : !llvm.struct<(ptr<i8>, i32)>
  llvm.return %0 : i32
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
  // expected-note@+1 {{global addresses expected as operand to bitcast used in clauses for landingpad}}
  %2 = llvm.bitcast %1 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  %3 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1: // pred: ^bb0
  llvm.return %0 : i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{constant clauses expected}}
  %5 = llvm.landingpad (catch %2 : !llvm.ptr<i8>) : !llvm.struct<(ptr<i8>, i32)>
  llvm.return %0 : i32
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1: // pred: ^bb0
  llvm.return %0 : i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{landingpad instruction expects at least one clause or cleanup attribute}}
  %2 = llvm.landingpad : !llvm.struct<(ptr<i8>, i32)>
  llvm.return %0 : i32
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0 } {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1: // pred: ^bb0
  llvm.return %0 : i32
^bb2: // pred: ^bb0
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr<i8>, i32)>
  // expected-error@+1 {{'llvm.resume' op expects landingpad value as operand}}
  llvm.resume %0 : i32
}

// -----

llvm.func @foo(i32) -> i32

llvm.func @caller(%arg0: i32) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1: // pred: ^bb0
  llvm.return %0 : i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{llvm.landingpad needs to be in a function with a personality}}
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr<i8>, i32)>
  llvm.resume %2 : !llvm.struct<(ptr<i8>, i32)>
}

// -----

func @invalid_ordering_in_fence() {
  // expected-error @+1 {{can be given only acquire, release, acq_rel, and seq_cst orderings}}
  llvm.fence syncscope("agent") monotonic
}

// -----

// expected-error @+1 {{invalid data layout descriptor}}
module attributes {llvm.data_layout = "#vjkr32"} {
  func @invalid_data_layout()
}

// -----

func @switch_wrong_number_of_weights(%arg0 : i32) {
  // expected-error@+1 {{expects number of branch weights to match number of successors: 3 vs 2}}
  llvm.switch %arg0, ^bb1 [
    42: ^bb2(%arg0, %arg0 : i32, i32)
  ] {branch_weights = dense<[13, 17, 19]> : vector<3xi32>}

^bb1: // pred: ^bb0
  llvm.return

^bb2(%1: i32, %2: i32): // pred: ^bb0
  llvm.return
}
