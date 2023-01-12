// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// expected-error@+1{{alignment attribute is not a power of 2}}
llvm.mlir.global private @invalid_global_alignment(42 : i64) {alignment = 63} : i64

// -----

llvm.func @ctor() {
  llvm.return
}

// expected-error@+1{{mismatch between the number of ctors and the number of priorities}}
llvm.mlir.global_ctors {ctors = [@ctor], priorities = []}

// -----

llvm.func @dtor() {
  llvm.return
}

// expected-error@+1{{mismatch between the number of dtors and the number of priorities}}
llvm.mlir.global_dtors {dtors = [@dtor], priorities = [0 : i32, 32767 : i32]}

// -----

// expected-error@+1{{'ctor' does not reference a valid LLVM function}}
llvm.mlir.global_ctors {ctors = [@ctor], priorities = [0 : i32]}

// -----

llvm.func @dtor()

// expected-error@+1{{'dtor' does not have a definition}}
llvm.mlir.global_dtors {dtors = [@dtor], priorities = [0 : i32]}

// -----

// expected-error@+1{{expected llvm.noalias argument attribute to be a unit attribute}}
func.func @invalid_noalias(%arg0: i32 {llvm.noalias = 3}) {
  "llvm.return"() : () -> ()
}

// -----

// expected-error@+1{{llvm.align argument attribute of non integer type}}
func.func @invalid_align(%arg0: i32 {llvm.align = "foo"}) {
  "llvm.return"() : () -> ()
}

////////////////////////////////////////////////////////////////////////////////

// Check that parser errors are properly produced and do not crash the compiler.

// -----

func.func @icmp_non_string(%arg0 : i32, %arg1 : i16) {
  // expected-error@+1 {{invalid kind of attribute specified}}
  llvm.icmp 42 %arg0, %arg0 : i32
  return
}

// -----

func.func @icmp_wrong_string(%arg0 : i32, %arg1 : i16) {
  // expected-error@+1 {{'foo' is an incorrect value of the 'predicate' attribute}}
  llvm.icmp "foo" %arg0, %arg0 : i32
  return
}

// -----

func.func @alloca_missing_input_result_type(%size : i64) {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : () -> ()
}

// -----

func.func @alloca_missing_input_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : () -> (!llvm.ptr<i32>)
}

// -----

func.func @alloca_missing_result_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : (i64) -> ()
}

// -----

func.func @alloca_non_function_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x i32 : !llvm.ptr<i32>
}

// -----

func.func @alloca_non_integer_alignment() {
  // expected-error@+1 {{expected integer alignment}}
  llvm.alloca %size x i32 {alignment = 3.0} : !llvm.ptr<i32>
}

// -----

func.func @alloca_opaque_ptr_no_type(%sz : i64) {
  // expected-error@below {{expected 'elem_type' attribute if opaque pointer type is used}}
  "llvm.alloca"(%sz) : (i64) -> !llvm.ptr
}

// -----

func.func @alloca_ptr_type_attr_non_opaque_ptr(%sz : i64) {
  // expected-error@below {{unexpected 'elem_type' attribute when non-opaque pointer type is used}}
  "llvm.alloca"(%sz) { elem_type = i32 } : (i64) -> !llvm.ptr<i32>
}

// -----

func.func @gep_missing_input_result_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{2 operands present, but expected 0}}
  llvm.getelementptr %base[%pos] : () -> ()
}

// -----

func.func @gep_missing_input_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{2 operands present, but expected 0}}
  llvm.getelementptr %base[%pos] : () -> (!llvm.ptr<f32>)
}

// -----

func.func @gep_missing_result_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{op requires one result}}
  llvm.getelementptr %base[%pos] : (!llvm.ptr<f32>, i64) -> ()
}

// -----

func.func @gep_non_function_type(%pos : i64, %base : !llvm.ptr<f32>) {
  // expected-error@+1 {{invalid kind of type specified}}
  llvm.getelementptr %base[%pos] : !llvm.ptr<f32>
}

// -----

func.func @gep_too_few_dynamic(%base : !llvm.ptr<f32>) {
  // expected-error@+1 {{expected as many dynamic indices as specified in 'rawConstantIndices'}}
  %1 = "llvm.getelementptr"(%base) {rawConstantIndices = array<i32: -2147483648>} : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
}

// -----

func.func @load_non_llvm_type(%foo : memref<f32>) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.load %foo : memref<f32>
}

// -----

func.func @load_non_ptr_type(%foo : f32) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.load %foo : f32
}

// -----

func.func @store_non_llvm_type(%foo : memref<f32>, %bar : f32) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.store %bar, %foo : memref<f32>
}

// -----

func.func @store_non_ptr_type(%foo : f32, %bar : f32) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.store %bar, %foo : f32
}

// -----

func.func @store_malformed_elem_type(%foo: !llvm.ptr, %bar: f32) {
  // expected-error@+1 {{expected non-function type}}
  llvm.store %bar, %foo : !llvm.ptr, "f32"
}

// -----

func.func @call_non_function_type(%callee : !llvm.func<i8 (i8)>, %arg : i8) {
  // expected-error@+1 {{expected function type}}
  llvm.call %callee(%arg) : !llvm.func<i8 (i8)>
  llvm.return
}

// -----

func.func @invalid_call() {
  // expected-error@+1 {{'llvm.call' op must have either a `callee` attribute or at least an operand}}
  "llvm.call"() : () -> ()
  llvm.return
}

// -----

func.func @call_non_function_type(%callee : !llvm.func<i8 (i8)>, %arg : i8) {
  // expected-error@+1 {{expected function type}}
  llvm.call %callee(%arg) : !llvm.func<i8 (i8)>
  llvm.return
}

// -----

func.func @call_unknown_symbol() {
  // expected-error@+1 {{'llvm.call' op 'missing_callee' does not reference a symbol in the current scope}}
  llvm.call @missing_callee() : () -> ()
  llvm.return
}

// -----

func.func private @standard_func_callee()

func.func @call_non_llvm() {
  // expected-error@+1 {{'llvm.call' op 'standard_func_callee' does not reference a valid LLVM function}}
  llvm.call @standard_func_callee() : () -> ()
  llvm.return
}

// -----

func.func @call_non_llvm_indirect(%arg0 : tensor<*xi32>) {
  // expected-error@+1 {{'llvm.call' op operand #0 must be LLVM dialect-compatible type}}
  "llvm.call"(%arg0) : (tensor<*xi32>) -> ()
  llvm.return
}

// -----

llvm.func @callee_func(i8) -> ()

func.func @callee_arg_mismatch(%arg0 : i32) {
  // expected-error@+1 {{'llvm.call' op operand type mismatch for operand 0: 'i32' != 'i8'}}
  llvm.call @callee_func(%arg0) : (i32) -> ()
  llvm.return
}

// -----

func.func @indirect_callee_arg_mismatch(%arg0 : i32, %callee : !llvm.ptr<func<void(i8)>>) {
  // expected-error@+1 {{'llvm.call' op operand type mismatch for operand 0: 'i32' != 'i8'}}
  "llvm.call"(%callee, %arg0) : (!llvm.ptr<func<void(i8)>>, i32) -> ()
  llvm.return
}

// -----

llvm.func @callee_func() -> (i8)

func.func @callee_return_mismatch() {
  // expected-error@+1 {{'llvm.call' op result type mismatch: 'i32' != 'i8'}}
  %res = llvm.call @callee_func() : () -> (i32)
  llvm.return
}

// -----

func.func @indirect_callee_return_mismatch(%callee : !llvm.ptr<func<i8()>>) {
  // expected-error@+1 {{'llvm.call' op result type mismatch: 'i32' != 'i8'}}
  "llvm.call"(%callee) : (!llvm.ptr<func<i8()>>) -> (i32)
  llvm.return
}

// -----

func.func @call_too_many_results(%callee : () -> (i32,i32)) {
  // expected-error@+1 {{expected function with 0 or 1 result}}
  llvm.call %callee() : () -> (i32, i32)
  llvm.return
}

// -----

func.func @call_non_llvm_result(%callee : () -> (tensor<*xi32>)) {
  // expected-error@+1 {{expected result to have LLVM type}}
  llvm.call %callee() : () -> (tensor<*xi32>)
  llvm.return
}

// -----

func.func @call_non_llvm_input(%callee : (tensor<*xi32>) -> (), %arg : tensor<*xi32>) {
  // expected-error@+1 {{expected LLVM types as inputs}}
  llvm.call %callee(%arg) : (tensor<*xi32>) -> ()
  llvm.return
}

// -----

llvm.func @void_func_result(%arg0: i32) {
  // expected-error@below {{expected no operands}}
  // expected-note@above {{when returning from function}}
  llvm.return %arg0: i32
}

// -----

llvm.func @non_void_func_no_result() -> i32 {
  // expected-error@below {{expected 1 operand}}
  // expected-note@above {{when returning from function}}
  llvm.return
}

// -----

llvm.func @func_result_mismatch(%arg0: f32) -> i32 {
  // expected-error@below {{mismatching result types}}
  // expected-note@above {{when returning from function}}
  llvm.return %arg0 : f32
}

// -----

func.func @constant_wrong_type() {
  // expected-error@+1 {{only supports integer, float, string or elements attributes}}
  llvm.mlir.constant(@constant_wrong_type) : !llvm.ptr<func<void ()>>
}

// -----

func.func @constant_wrong_type_string() {
  // expected-error@below {{expected array type of 3 i8 elements for the string constant}}
  llvm.mlir.constant("foo") : !llvm.ptr<i8>
}

// -----

llvm.func @array_attribute_one_element() -> !llvm.struct<(f64, f64)> {
  // expected-error @+1 {{expected array attribute with two elements, representing a complex constant}}
  %0 = llvm.mlir.constant([1.0 : f64]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

llvm.func @array_attribute_two_different_types() -> !llvm.struct<(f64, f64)> {
  // expected-error @+1 {{expected array attribute with two elements of the same type}}
  %0 = llvm.mlir.constant([1.0 : f64, 1.0 : f32]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

llvm.func @struct_wrong_attribute_type() -> !llvm.struct<(f64, f64)> {
  // expected-error @+1 {{expected array attribute with two elements, representing a complex constant}}
  %0 = llvm.mlir.constant(1.0 : f64) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

llvm.func @struct_one_element() -> !llvm.struct<(f64)> {
  // expected-error @+1 {{expected struct type with two elements of the same type, the type of a complex constant}}
  %0 = llvm.mlir.constant([1.0 : f64, 1.0 : f64]) : !llvm.struct<(f64)>
  llvm.return %0 : !llvm.struct<(f64)>
}

// -----

llvm.func @struct_two_different_elements() -> !llvm.struct<(f64, f32)> {
  // expected-error @+1 {{expected struct type with two elements of the same type, the type of a complex constant}}
  %0 = llvm.mlir.constant([1.0 : f64, 1.0 : f64]) : !llvm.struct<(f64, f32)>
  llvm.return %0 : !llvm.struct<(f64, f32)>
}

// -----

llvm.func @struct_wrong_element_types() -> !llvm.struct<(!llvm.array<2 x f64>, !llvm.array<2 x f64>)> {
  // expected-error @+1 {{expected struct element types to be floating point type or integer type}}
  %0 = llvm.mlir.constant([dense<[1.0, 1.0]> : tensor<2xf64>, dense<[1.0, 1.0]> : tensor<2xf64>]) : !llvm.struct<(!llvm.array<2 x f64>, !llvm.array<2 x f64>)>
  llvm.return %0 : !llvm.struct<(!llvm.array<2 x f64>, !llvm.array<2 x f64>)>
}

// -----

func.func @insertvalue_non_llvm_type(%a : i32, %b : i32) {
  // expected-error@+2 {{expected LLVM IR Dialect type}}
  llvm.insertvalue %a, %b[0] : tensor<*xi32>
}

// -----

func.func @insertvalue_struct_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm.struct<(i32)>
}

// -----

func.func @insertvalue_array_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm.array<1 x i32>
}

// -----

func.func @insertvalue_wrong_nesting() {
  // expected-error@+2 {{expected LLVM IR structure/array type}}
  llvm.insertvalue %a, %b[0,0] : !llvm.struct<(i32)>
}

// -----

func.func @insertvalue_invalid_type(%a : !llvm.array<1 x i32>) -> !llvm.array<1 x i32> {
  // expected-error@+1 {{'llvm.insertvalue' op Type mismatch: cannot insert '!llvm.array<1 x i32>' into '!llvm.array<1 x i32>'}}
  %b = "llvm.insertvalue"(%a, %a) {position = array<i64: 0>} : (!llvm.array<1 x i32>, !llvm.array<1 x i32>) -> !llvm.array<1 x i32>
  return %b : !llvm.array<1 x i32>
}

// -----

func.func @extractvalue_invalid_type(%a : !llvm.array<4 x vector<8xf32>>) -> !llvm.array<4 x vector<8xf32>> {
  // expected-error@+1 {{'llvm.extractvalue' op Type mismatch: extracting from '!llvm.array<4 x vector<8xf32>>' should produce 'vector<8xf32>' but this op returns '!llvm.array<4 x vector<8xf32>>'}}
  %b = "llvm.extractvalue"(%a) {position = array<i64: 1>}
            : (!llvm.array<4 x vector<8xf32>>) -> !llvm.array<4 x vector<8xf32>>
  return %b : !llvm.array<4 x vector<8xf32>>
}


// -----

func.func @extractvalue_non_llvm_type(%a : i32, %b : tensor<*xi32>) {
  // expected-error@+2 {{expected LLVM IR Dialect type}}
  llvm.extractvalue %b[0] : tensor<*xi32>
}
// -----

func.func @extractvalue_struct_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm.struct<(i32)>
}

// -----

func.func @extractvalue_array_out_of_bounds() {
  // expected-error@+2 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm.array<1 x i32>
}

// -----

func.func @extractvalue_wrong_nesting() {
  // expected-error@+2 {{expected LLVM IR structure/array type}}
  llvm.extractvalue %b[0,0] : !llvm.struct<(i32)>
}

// -----

func.func @invalid_vector_type_1(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{'vector' must be LLVM dialect-compatible vector}}
  %0 = llvm.extractelement %arg2[%arg1 : i32] : f32
}

// -----

func.func @invalid_vector_type_2(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{'vector' must be LLVM dialect-compatible vector}}
  %0 = llvm.insertelement %arg2, %arg2[%arg1 : i32] : f32
}

// -----

func.func @invalid_vector_type_3(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // expected-error@+2 {{expected an LLVM compatible vector type}}
  %0 = llvm.shufflevector %arg2, %arg2 [0, 0, 0, 0, 7] : f32
}

// -----

func.func @invalid_vector_type_4(%a : vector<4xf32>, %idx : i32) -> vector<4xf32> {
  // expected-error@+1 {{failed to verify that result type matches vector element type}}
  %b = "llvm.extractelement"(%a, %idx) : (vector<4xf32>, i32) -> vector<4xf32>
  return %b : vector<4xf32>
}

// -----

func.func @invalid_vector_type_5(%a : vector<4xf32>, %idx : i32) -> vector<4xf32> {
  // expected-error@+1 {{failed to verify that argument type matches vector element type}}
  %b = "llvm.insertelement"(%a, %a, %idx) : (vector<4xf32>, vector<4xf32>, i32) -> vector<4xf32>
  return %b : vector<4xf32>
}

// -----

func.func @null_non_llvm_type() {
  // expected-error@+1 {{'llvm.mlir.null' op result #0 must be LLVM pointer type, but got 'i32'}}
  llvm.mlir.null : i32
}

// -----

func.func @nvvm_invalid_shfl_pred_1(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> i32
}

// -----

func.func @nvvm_invalid_shfl_pred_2(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32)>
}

// -----

func.func @nvvm_invalid_shfl_pred_3(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i32)>
}

// -----

func.func @nvvm_invalid_mma_0(%a0 : f16, %a1 : f16,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{Could not match types for the A operands; expected one of 2xvector<2xf16> but got f16, f16}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {layoutA=#nvvm.mma_layout<row>, layoutB=#nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (f16, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func.func @nvvm_invalid_mma_1(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{Could not match allowed types for the result; expected one of !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> but got !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {layoutA=#nvvm.mma_layout<row>, layoutB=#nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f16)>
}

// -----

func.func @nvvm_invalid_mma_2(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                         %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // expected-error@+1 {{op requires attribute 'layoutA'}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {shape = #nvvm.shape<m = 8, n = 8, k = 4>}: (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

func.func @nvvm_invalid_mma_3(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                         %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                         %c0 : vector<2xf16>, %c1 : vector<2xf16>) {
  // expected-error@+1 {{unimplemented variant for MMA shape <8, 8, 16>}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1] {layoutA=#nvvm.mma_layout<row>, layoutB=#nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// -----

func.func @nvvm_invalid_mma_8(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // expected-error@+1 {{op requires b1Op attribute}}
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// -----

func.func @atomicrmw_expected_ptr(%f32 : f32) {
  // expected-error@+1 {{operand #0 must be LLVM pointer to floating point LLVM type or integer}}
  %0 = "llvm.atomicrmw"(%f32, %f32) {bin_op=11, ordering=1} : (f32, f32) -> f32
  llvm.return
}

// -----

func.func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr<f32>, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR element type for operand #0 to match type for operand #1}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %i32) {bin_op=11, ordering=1} : (!llvm.ptr<f32>, i32) -> f32
  llvm.return
}

// -----

func.func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr<f32>, %f32 : f32) {
  // expected-error@+1 {{expected LLVM IR result type to match type for operand #1}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %f32) {bin_op=11, ordering=1} : (!llvm.ptr<f32>, f32) -> i32
  llvm.return
}

// -----

func.func @atomicrmw_expected_float(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR floating point type}}
  %0 = llvm.atomicrmw fadd %i32_ptr, %i32 unordered : i32
  llvm.return
}

// -----

func.func @atomicrmw_unexpected_xchg_type(%i1_ptr : !llvm.ptr<i1>, %i1 : i1) {
  // expected-error@+1 {{unexpected LLVM IR type for 'xchg' bin_op}}
  %0 = llvm.atomicrmw xchg %i1_ptr, %i1 unordered : i1
  llvm.return
}

// -----

func.func @atomicrmw_expected_int(%f32_ptr : !llvm.ptr<f32>, %f32 : f32) {
  // expected-error@+1 {{expected LLVM IR integer type}}
  %0 = llvm.atomicrmw max %f32_ptr, %f32 unordered : f32
  llvm.return
}

// -----

func.func @cmpxchg_expected_ptr(%f32_ptr : !llvm.ptr<f32>, %f32 : f32) {
  // expected-error@+1 {{op operand #0 must be LLVM pointer to integer or LLVM pointer type}}
  %0 = "llvm.cmpxchg"(%f32, %f32, %f32) {success_ordering=2,failure_ordering=2} : (f32, f32, f32) -> !llvm.struct<(f32, i1)>
  llvm.return
}

// -----

func.func @cmpxchg_mismatched_operands(%i64_ptr : !llvm.ptr<i64>, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR element type for operand #0 to match type for all other operands}}
  %0 = "llvm.cmpxchg"(%i64_ptr, %i32, %i32) {success_ordering=2,failure_ordering=2} : (!llvm.ptr<i64>, i32, i32) -> !llvm.struct<(i32, i1)>
  llvm.return
}

// -----

func.func @cmpxchg_unexpected_type(%i1_ptr : !llvm.ptr<i1>, %i1 : i1) {
  // expected-error@+1 {{unexpected LLVM IR type}}
  %0 = llvm.cmpxchg %i1_ptr, %i1, %i1 monotonic monotonic : i1
  llvm.return
}

// -----

func.func @cmpxchg_at_least_monotonic_success(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 unordered monotonic : i32
  llvm.return
}

// -----

func.func @cmpxchg_at_least_monotonic_failure(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 monotonic unordered : i32
  llvm.return
}

// -----

func.func @cmpxchg_failure_release(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel release : i32
  llvm.return
}

// -----

func.func @cmpxchg_failure_acq_rel(%i32_ptr : !llvm.ptr<i32>, %i32 : i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel acq_rel : i32
  llvm.return
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @bad_landingpad(%arg0: !llvm.ptr<ptr<i8>>) -> i32 attributes { personality = @__gxx_personality_v0} {
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

func.func @invalid_ordering_in_fence() {
  // expected-error @+1 {{can be given only acquire, release, acq_rel, and seq_cst orderings}}
  llvm.fence syncscope("agent") monotonic
}

// -----

// expected-error @+1 {{invalid data layout descriptor}}
module attributes {llvm.data_layout = "#vjkr32"} {
  func.func @invalid_data_layout()
}

// -----

func.func @switch_wrong_number_of_weights(%arg0 : i32) {
  // expected-error@+1 {{expects number of branch weights to match number of successors: 3 vs 2}}
  llvm.switch %arg0 : i32, ^bb1 [
    42: ^bb2(%arg0, %arg0 : i32, i32)
  ] {branch_weights = dense<[13, 17, 19]> : vector<3xi32>}

^bb1: // pred: ^bb0
  llvm.return

^bb2(%1: i32, %2: i32): // pred: ^bb0
  llvm.return
}

// -----

// expected-error@below {{expected zero value for 'common' linkage}}
llvm.mlir.global common @non_zero_global_common_linkage(42 : i32) : i32

// -----

// expected-error@below {{expected zero value for 'common' linkage}}
llvm.mlir.global common @non_zero_compound_global_common_linkage(dense<[0, 0, 0, 1, 0]> : vector<5xi32>) : !llvm.array<5 x i32>

// -----

// expected-error@below {{expected array type for 'appending' linkage}}
llvm.mlir.global appending @non_array_type_global_appending_linkage() : i32

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{expected 'llvm.loop' to be a dictionary attribute}}
      llvm.br ^bb4 {llvm.loop = "test"}
    ^bb4:
      llvm.return
  }
}

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{expected 'parallel_access' to be an array attribute}}
      llvm.br ^bb4 {llvm.loop = {parallel_access = "loop"}}
    ^bb4:
      llvm.return
  }
}

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{expected '"loop"' to be a symbol reference}}
      llvm.br ^bb4 {llvm.loop = {parallel_access = ["loop"]}}
    ^bb4:
      llvm.return
  }
}

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{expected '@func1' to reference a metadata op}}
      llvm.br ^bb4 {llvm.loop = {parallel_access = [@func1]}}
    ^bb4:
      llvm.return
  }
  llvm.func @func1() {
    llvm.return
  }
}

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{expected '@metadata' to reference an access_group op}}
      llvm.br ^bb4 {llvm.loop = {parallel_access = [@metadata]}}
    ^bb4:
      llvm.return
  }
  llvm.metadata @metadata {
  }
}

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{expected 'options' to be a `loopopts` attribute}}
      llvm.br ^bb4 {llvm.loop = {options = "name"}}
    ^bb4:
      llvm.return
  }
}

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{unknown loop option: name}}
      llvm.br ^bb4 {llvm.loop = {options = #llvm.loopopts<name>}}
    ^bb4:
      llvm.return
  }
}

// -----

module {
  llvm.func @loopOptions() {
      // expected-error@below {{loop option present twice}}
      llvm.br ^bb4 {llvm.loop = {options = #llvm.loopopts<disable_licm = true, disable_licm = true>}}
    ^bb4:
      llvm.return
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute}}
      %0 = llvm.load %arg0 { "access_groups" = "test" } : !llvm.ptr<i32>
      llvm.return
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{expected '@func1' to specify a fully qualified reference}}
      %0 = llvm.load %arg0 { "access_groups" = [@func1] } : !llvm.ptr<i32>
      llvm.return
  }
  llvm.func @func1() {
    llvm.return
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{expected '@accessGroups::@group1' to reference a metadata op}}
      %0 = llvm.load %arg0 { "access_groups" = [@accessGroups::@group1] } : !llvm.ptr<i32>
      llvm.return
  }
  llvm.metadata @metadata {
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{expected '@metadata::@group1' to be a valid reference}}
      %0 = llvm.load %arg0 { "access_groups" = [@metadata::@group1] } : !llvm.ptr<i32>
      llvm.return
  }
  llvm.metadata @metadata {
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{expected '@metadata::@scope' to resolve to a llvm.access_group}}
      %0 = llvm.load %arg0 { "access_groups" = [@metadata::@scope] } : !llvm.ptr<i32>
      llvm.return
  }
  llvm.metadata @metadata {
    llvm.alias_scope_domain @domain
    llvm.alias_scope @scope { domain = @domain }
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{attribute 'alias_scopes' failed to satisfy constraint: symbol ref array attribute}}
      %0 = llvm.load %arg0 { "alias_scopes" = "test" } : !llvm.ptr<i32>
      llvm.return
  }
}

// -----

module {
  llvm.func @accessGroups(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{attribute 'noalias_scopes' failed to satisfy constraint: symbol ref array attribute}}
      %0 = llvm.load %arg0 { "noalias_scopes" = "test" } : !llvm.ptr<i32>
      llvm.return
  }
}

// -----

module {
  llvm.func @aliasScope(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{expected '@metadata::@group' to resolve to a llvm.alias_scope}}
      %0 = llvm.load %arg0 { "alias_scopes" = [@metadata::@group] } : !llvm.ptr<i32>
      llvm.return
  }
  llvm.metadata @metadata {
    llvm.access_group @group
  }
}

// -----

module {
  llvm.func @aliasScope(%arg0 : !llvm.ptr<i32>) {
      // expected-error@below {{expected '@metadata::@group' to resolve to a llvm.alias_scope}}
      %0 = llvm.load %arg0 { "noalias_scopes" = [@metadata::@group] } : !llvm.ptr<i32>
      llvm.return
  }
  llvm.metadata @metadata {
    llvm.access_group @group
  }
}

// -----

llvm.func @wmmaLoadOp_invalid_mem_space(%arg0: !llvm.ptr<i32, 5>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected source pointer in memory space 0, 1, 3}}
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<i32, 5>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_AOp(%arg0: !llvm.ptr<i32, 3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 8 elements of type 'vector<2xf16>'}}
  %0 = nvvm.wmma.load %arg0, %arg1
  {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  : (!llvm.ptr<i32, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_BOp(%arg0: !llvm.ptr<i32, 3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 8 elements of type 'vector<2xf16>'}}
 %0 = nvvm.wmma.load %arg0, %arg1
 {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<b>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
 : (!llvm.ptr<i32, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_COp(%arg0: !llvm.ptr<i32, 3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 4 elements of type 'vector<2xf16>'}}
 %0 = nvvm.wmma.load %arg0, %arg1
   {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<c>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
   : (!llvm.ptr<i32, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// -----

llvm.func @wmmaStoreOp_invalid_mem_space(%arg0: !llvm.ptr<i32, 5>, %arg1: i32,
                            %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                            %arg4: vector<2 x f16>, %arg5: vector<2 xf16>) {
  // expected-error@+1 {{'nvvm.wmma.store' op expected operands to be a source pointer in memory space 0, 1, 3}}
  nvvm.wmma.store %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : !llvm.ptr<i32, 5>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_operands(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2 x f16>,
                        %arg16: vector<2 x f16>, %arg17: vector<2 x f16>,
                        %arg18: vector<2 x f16>) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected 20 arguments}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)
      -> !llvm.struct<(vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_results(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2 x f16>,
                        %arg16: vector<2 x f16>, %arg17: vector<2 x f16>,
                        %arg18: vector<2 x f16>, %arg19: vector<2 x f16>) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected destination type is a structure of 4 elements of type 'vector<2xf16>'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)
      -> !llvm.struct<(vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)>  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_ab_operands(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: f32,
                        %arg16: f32, %arg17: f32, %arg18: f32, %arg19: f32,
                        %arg20: f32, %arg21: f32, %arg22: f32, %arg23: f32) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected argument 15 to be of type 'vector<2xf16>'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f32>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_c_operand(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2xf16>,
                        %arg16: f32, %arg17: f32, %arg18: f32, %arg19: f32,
                        %arg20: f32, %arg21: f32, %arg22: f32, %arg23: vector<2xf16>) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected argument 23 to be of type 'f32'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f32>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return
}

// -----

llvm.func @gpu_wmma_mma_op_invalid_result(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2xf16>,
                        %arg16: f32, %arg17: f32, %arg18: f32, %arg19: f32,
                        %arg20: f32, %arg21: f32, %arg22: f32, %arg23: f32) {
  // expected-error@+1 {{'nvvm.wmma.mma' op expected destination type is a structure of 8 elements of type 'f32'}}
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f32>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr<i32>) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected source pointer in memory space 3}}
  %l = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32>) -> i32
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr<i32, 3>) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected num attribute to be 1, 2 or 4}}
  %l = nvvm.ldmatrix %arg0 {num = 3 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> i32
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr<i32, 3>) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected destination type is i32}}
  %l = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32)>
  llvm.return
}

// -----

llvm.func @wmmald_matrix(%arg0: !llvm.ptr<i32, 3>) {
  // expected-error@+1 {{'nvvm.ldmatrix' op expected destination type is a structure of 4 elements of type i32}}
  %l = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32)>
  llvm.return
}

// -----

llvm.func @caller() {
  // expected-error @below {{expected function call to produce a value}}
  llvm.call @callee() : () -> ()
  llvm.return
}

llvm.func @callee() -> i32

// -----

llvm.func @caller() {
  // expected-error @below {{calling function with void result must not produce values}}
  %0 = llvm.call @callee() : () -> i32
  llvm.return
}

llvm.func @callee() -> ()

// -----

llvm.func @caller() {
  // expected-error @below {{expected function with 0 or 1 result}}
  %0:2 = llvm.call @callee() : () -> (i32, f32)
  llvm.return
}

llvm.func @callee() -> !llvm.struct<(i32, f32)>

// -----

func.func @bitcast(%arg0: vector<2x3xf32>) {
  // expected-error @below {{op operand #0 must be LLVM-compatible non-aggregate type}}
  llvm.bitcast %arg0 : vector<2x3xf32> to vector<2x3xi32>
  return
}

// -----

func.func @cp_async(%arg0: !llvm.ptr<i8, 3>, %arg1: !llvm.ptr<i8, 1>) {
  // expected-error @below {{expected byte size to be either 4, 8 or 16.}}
  nvvm.cp.async.shared.global %arg0, %arg1, 32
  return
}

// -----

func.func @cp_async(%arg0: !llvm.ptr<i8, 3>, %arg1: !llvm.ptr<i8, 1>) {
  // expected-error @below {{bypass l1 is only support for 16 bytes copy.}}
  nvvm.cp.async.shared.global %arg0, %arg1, 8 {bypass_l1}
  return
}

// -----

func.func @gep_struct_variable(%arg0: !llvm.ptr<struct<(i32)>>, %arg1: i32, %arg2: i32) {
  // expected-error @below {{op expected index 1 indexing a struct to be constant}}
  llvm.getelementptr %arg0[%arg1, %arg1] : (!llvm.ptr<struct<(i32)>>, i32, i32) -> !llvm.ptr<i32>
  return
}

// -----

func.func @gep_out_of_bounds(%ptr: !llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, %idx: i64) {
  // expected-error @below {{index 2 indexing a struct is out of bounds}}
  llvm.getelementptr %ptr[%idx, 1, 3] : (!llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, i64) -> !llvm.ptr<i32>
  return
}

// -----

func.func @non_splat_shuffle_on_scalable_vector(%arg0: vector<[4]xf32>) {
  // expected-error@below {{expected a splat operation for scalable vectors}}
  %0 = llvm.shufflevector %arg0, %arg0 [0, 0, 0, 1] : vector<[4]xf32>
  return
}

// -----

llvm.mlir.global internal @side_effecting_global() : !llvm.struct<(i8)> {
  %0 = llvm.mlir.constant(1 : i64) : i64
  // expected-error@below {{ops with side effects not allowed in global initializers}}
  %1 = llvm.alloca %0 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr<struct<(i8)>>
  %2 = llvm.load %1 : !llvm.ptr<struct<(i8)>>
  llvm.return %2 : !llvm.struct<(i8)>
}

// -----

// expected-error@+1 {{'llvm.struct_attrs' is permitted only in argument or result attributes}}
func.func @struct_attrs_in_op() attributes {llvm.struct_attrs = []} {
  return
}

// -----

// expected-error@+1 {{expected 'llvm.struct_attrs' to annotate '!llvm.struct' or '!llvm.ptr<struct<...>>'}}
func.func @invalid_struct_attr_arg_type(%arg0 : i32 {llvm.struct_attrs = []}) {
    return
}

// -----

// expected-error@+1 {{expected 'llvm.struct_attrs' to annotate '!llvm.struct' or '!llvm.ptr<struct<...>>'}}
func.func @invalid_struct_attr_pointer_arg_type(%arg0 : !llvm.ptr<i32> {llvm.struct_attrs = []}) {
    return
}

// -----

// expected-error@+1 {{expected 'llvm.struct_attrs' to be an array attribute}}
func.func @invalid_arg_struct_attr_value(%arg0 : !llvm.struct<(i32)> {llvm.struct_attrs = {}}) {
    return
}

// -----

// expected-error@+1 {{size of 'llvm.struct_attrs' must match the size of the annotated '!llvm.struct'}}
func.func @invalid_arg_struct_attr_size(%arg0 : !llvm.struct<(i32)> {llvm.struct_attrs = []}) {
    return
}

// -----

// expected-error@+1 {{expected 'llvm.struct_attrs' to annotate '!llvm.struct' or '!llvm.ptr<struct<...>>'}}
func.func @invalid_struct_attr_res_type(%arg0 : i32) -> (i32 {llvm.struct_attrs = []}) {
  return %arg0 : i32
}

// -----

// expected-error@+1 {{expected 'llvm.struct_attrs' to annotate '!llvm.struct' or '!llvm.ptr<struct<...>>'}}
func.func @invalid_struct_attr_pointer_res_type(%arg0 : !llvm.ptr<i32>) -> (!llvm.ptr<i32> {llvm.struct_attrs = []}) {
    return %arg0 : !llvm.ptr<i32>
}

// -----

// expected-error@+1 {{expected 'llvm.struct_attrs' to be an array attribute}}
func.func @invalid_res_struct_attr_value(%arg0 : !llvm.struct<(i32)>) -> (!llvm.struct<(i32)> {llvm.struct_attrs = {}}) {
    return %arg0 : !llvm.struct<(i32)>
}

// -----

// expected-error@+1 {{size of 'llvm.struct_attrs' must match the size of the annotated '!llvm.struct'}}
func.func @invalid_res_struct_attr_size(%arg0 : !llvm.struct<(i32)>) -> (!llvm.struct<(i32)> {llvm.struct_attrs = []}) {
    return %arg0 : !llvm.struct<(i32)>
}

// -----

func.func @insert_vector_invalid_source_vector_size(%arg0 : vector<16385xi8>, %arg1 : vector<[16]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.insert %arg0, %arg1[0] : vector<16385xi8> into vector<[16]xi8>
}

// -----

func.func @insert_vector_invalid_dest_vector_size(%arg0 : vector<16xi8>, %arg1 : vector<[16385]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.insert %arg0, %arg1[0] : vector<16xi8> into vector<[16385]xi8>
}

// -----

func.func @insert_scalable_into_fixed_length_vector(%arg0 : vector<[8]xf32>, %arg1 : vector<16xf32>) {
  // expected-error@+1 {{op failed to verify that it is not inserting scalable into fixed-length vectors.}}
  %0 = llvm.intr.vector.insert %arg0, %arg1[0] : vector<[8]xf32> into vector<16xf32>
}

// -----

func.func @extract_vector_invalid_source_vector_size(%arg0 : vector<[16385]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.extract %arg0[0] : vector<16xi8> from vector<[16385]xi8>
}

// -----

func.func @extract_vector_invalid_result_vector_size(%arg0 : vector<[16]xi8>) {
  // expected-error@+1 {{op failed to verify that vectors are not bigger than 2^17 bits.}}
  %0 = llvm.intr.vector.extract %arg0[0] : vector<16385xi8> from vector<[16]xi8>
}

// -----

func.func @extract_scalable_from_fixed_length_vector(%arg0 : vector<16xf32>) {
  // expected-error@+1 {{op failed to verify that it is not extracting scalable from fixed-length vectors.}}
  %0 = llvm.intr.vector.extract %arg0[0] : vector<[8]xf32> from vector<16xf32>
}
