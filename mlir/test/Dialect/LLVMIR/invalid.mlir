// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// expected-error@+1{{llvm.noalias argument attribute of non boolean type}}
func @invalid_noalias(%arg0: !llvm.i32 {llvm.noalias = 3}) {
  "llvm.return"() : () -> ()
}

// -----

// expected-error@+1{{llvm.align argument attribute of non integer type}}
func @invalid_align(%arg0: !llvm.i32 {llvm.align = "foo"}) {
  "llvm.return"() : () -> ()
}

////////////////////////////////////////////////////////////////////////////////

// Check that parser errors are properly produced and do not crash the compiler.

// -----

func @icmp_non_string(%arg0 : !llvm.i32, %arg1 : !llvm.i16) {
  // expected-error@+1 {{invalid kind of attribute specified}}
  llvm.icmp 42 %arg0, %arg0 : !llvm.i32
  return
}

// -----

func @icmp_wrong_string(%arg0 : !llvm.i32, %arg1 : !llvm.i16) {
  // expected-error@+1 {{'foo' is an incorrect value of the 'predicate' attribute}}
  llvm.icmp "foo" %arg0, %arg0 : !llvm.i32
  return
}

// -----

func @alloca_missing_input_result_type(%size : !llvm.i64) {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : () -> ()
}

// -----

func @alloca_missing_input_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : () -> (!llvm.ptr<i32>)
}

// -----

func @alloca_mising_result_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : (!llvm.i64) -> ()
}

// -----

func @alloca_non_function_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : !llvm.ptr<i32>
}

// -----

func @alloca_nonpositive_alignment(%size : !llvm.i64) {
  // expected-error@+1 {{expected positive alignment}}
  llvm.alloca %size x !llvm.i32 {alignment = -1} : (!llvm.i64) -> (!llvm.ptr<i32>)
}

// -----

func @gep_missing_input_result_type(%pos : !llvm.i64, %base : !llvm.ptr<float>) {
  // expected-error@+1 {{2 operands present, but expected 0}}
  llvm.getelementptr %base[%pos] : () -> ()
}

// -----

func @gep_missing_input_type(%pos : !llvm.i64, %base : !llvm.ptr<float>) {
  // expected-error@+1 {{2 operands present, but expected 0}}
  llvm.getelementptr %base[%pos] : () -> (!llvm.ptr<float>)
}

// -----

func @gep_missing_result_type(%pos : !llvm.i64, %base : !llvm.ptr<float>) {
  // expected-error@+1 {{op requires one result}}
  llvm.getelementptr %base[%pos] : (!llvm.ptr<float>, !llvm.i64) -> ()
}

// -----

func @gep_non_function_type(%pos : !llvm.i64, %base : !llvm.ptr<float>) {
  // expected-error@+1 {{invalid kind of type specified}}
  llvm.getelementptr %base[%pos] : !llvm.ptr<float>
}

// -----

func @load_non_llvm_type(%foo : memref<f32>) {
  // expected-error@+1 {{expected LLVM IR dialect type}}
  llvm.load %foo : memref<f32>
}

// -----

func @load_non_ptr_type(%foo : !llvm.float) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.load %foo : !llvm.float
}

// -----

func @store_non_llvm_type(%foo : memref<f32>, %bar : !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect type}}
  llvm.store %bar, %foo : memref<f32>
}

// -----

func @store_non_ptr_type(%foo : !llvm.float, %bar : !llvm.float) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.store %bar, %foo : !llvm.float
}

// -----

func @call_non_function_type(%callee : !llvm.func<i8 (i8)>, %arg : !llvm.i8) {
  // expected-error@+1 {{expected function type}}
  llvm.call %callee(%arg) : !llvm.func<i8 (i8)>
}

// -----

func @call_too_many_results(%callee : () -> (i32,i32)) {
  // expected-error@+1 {{expected function with 0 or 1 result}}
  llvm.call %callee() : () -> (i32, i32)
}

// -----

func @call_non_llvm_result(%callee : () -> (i32)) {
  // expected-error@+1 {{expected result to have LLVM type}}
  llvm.call %callee() : () -> (i32)
}

// -----

func @call_non_llvm_input(%callee : (i32) -> (), %arg : i32) {
  // expected-error@+1 {{expected LLVM types as inputs}}
  llvm.call %callee(%arg) : (i32) -> ()
}

// -----

func @constant_wrong_type() {
  // expected-error@+1 {{only supports integer, float, string or elements attributes}}
  llvm.mlir.constant(@constant_wrong_type) : !llvm.ptr<func<void ()>>
}

// -----

func @insertvalue_non_llvm_type(%a : i32, %b : i32) {
  // expected-error@+1 {{expected LLVM IR Dialect type}}
  llvm.insertvalue %a, %b[0] : i32
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
  // expected-error@+1 {{expected wrapped LLVM IR structure/array type}}
  llvm.insertvalue %a, %b[0,0] : !llvm.struct<(i32)>
}

// -----

func @extractvalue_non_llvm_type(%a : i32, %b : i32) {
  // expected-error@+1 {{expected LLVM IR Dialect type}}
  llvm.extractvalue %b[0] : i32
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
  // expected-error@+1 {{expected wrapped LLVM IR structure/array type}}
  llvm.extractvalue %b[0,0] : !llvm.struct<(i32)>
}

// -----

// CHECK-LABEL: @invalid_vector_type_1
func @invalid_vector_type_1(%arg0: !llvm.vec<4 x float>, %arg1: !llvm.i32, %arg2: !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect vector type for operand #1}}
  %0 = llvm.extractelement %arg2[%arg1 : !llvm.i32] : !llvm.float
}

// -----

// CHECK-LABEL: @invalid_vector_type_2
func @invalid_vector_type_2(%arg0: !llvm.vec<4 x float>, %arg1: !llvm.i32, %arg2: !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect vector type for operand #1}}
  %0 = llvm.insertelement %arg2, %arg2[%arg1 : !llvm.i32] : !llvm.float
}

// -----

// CHECK-LABEL: @invalid_vector_type_3
func @invalid_vector_type_3(%arg0: !llvm.vec<4 x float>, %arg1: !llvm.i32, %arg2: !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect vector type for operand #1}}
  %0 = llvm.shufflevector %arg2, %arg2 [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : !llvm.float, !llvm.float
}

// -----

func @null_non_llvm_type() {
  // expected-error@+1 {{expected LLVM IR pointer type}}
  llvm.mlir.null : !llvm.i32
}

// -----

// CHECK-LABEL: @nvvm_invalid_shfl_pred_1
func @nvvm_invalid_shfl_pred_1(%arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32, %arg3 : !llvm.i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm.i32
}

// -----

// CHECK-LABEL: @nvvm_invalid_shfl_pred_2
func @nvvm_invalid_shfl_pred_2(%arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32, %arg3 : !llvm.i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm.struct<(i32)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_shfl_pred_3
func @nvvm_invalid_shfl_pred_3(%arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32, %arg3 : !llvm.i32) {
  // expected-error@+1 {{expected return type to be a two-element struct with i1 as the second element}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm.struct<(i32, i32)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_0
func @nvvm_invalid_mma_0(%a0 : !llvm.half, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{expected operands to be 4 <halfx2>s followed by either 4 <halfx2>s or 8 floats}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (!llvm.half, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm.struct<(float, float, float, float, float, float, float, float)>
  llvm.return %0 : !llvm.struct<(float, float, float, float, float, float, float, float)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_1
func @nvvm_invalid_mma_1(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{expected result type to be a struct of either 4 <halfx2>s or 8 floats}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (!llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm.struct<(float, float, float, float, float, float, float, half)>
  llvm.return %0 : !llvm.struct<(float, float, float, float, float, float, float, half)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_2
func @nvvm_invalid_mma_2(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{alayout and blayout attributes must be set to either "row" or "col"}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 : (!llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm.struct<(float, float, float, float, float, float, float, float)>
  llvm.return %0 : !llvm.struct<(float, float, float, float, float, float, float, float)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_3
func @nvvm_invalid_mma_3(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.vec<2 x half>, %c1 : !llvm.vec<2 x half>,
                         %c2 : !llvm.vec<2 x half>, %c3 : !llvm.vec<2 x half>) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3 {alayout="row", blayout="col"} : (!llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>) -> !llvm.struct<(float, float, float, float, float, float, float, float)>
  llvm.return %0 : !llvm.struct<(float, float, float, float, float, float, float, float)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_4
func @nvvm_invalid_mma_4(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (!llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
  llvm.return %0 : !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_5
func @nvvm_invalid_mma_5(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : (!llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm.struct<(float, float, float, float, float, float, float, float)>
  llvm.return %0 : !llvm.struct<(float, float, float, float, float, float, float, float)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_6
func @nvvm_invalid_mma_6(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{invalid kind of type specified}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : !llvm.struct<(float, float, float, float, float, float, float, float)>
  llvm.return %0 : !llvm.struct<(float, float, float, float, float, float, float, float)>
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_7
func @nvvm_invalid_mma_7(%a0 : !llvm.vec<2 x half>, %a1 : !llvm.vec<2 x half>,
                         %b0 : !llvm.vec<2 x half>, %b1 : !llvm.vec<2 x half>,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{op requires one result}}
  %0:2 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : (!llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.vec<2 x half>, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> (!llvm.struct<(float, float, float, float, float, float, float, float)>, !llvm.i32)
  llvm.return %0#0 : !llvm.struct<(float, float, float, float, float, float, float, float)>
}

// -----

// CHECK-LABEL: @atomicrmw_expected_ptr
func @atomicrmw_expected_ptr(%f32 : !llvm.float) {
  // expected-error@+1 {{expected LLVM IR pointer type for operand #0}}
  %0 = "llvm.atomicrmw"(%f32, %f32) {bin_op=11, ordering=1} : (!llvm.float, !llvm.float) -> !llvm.float
  llvm.return
}

// -----

// CHECK-LABEL: @atomicrmw_mismatched_operands
func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr<float>, %i32 : !llvm.i32) {
  // expected-error@+1 {{expected LLVM IR element type for operand #0 to match type for operand #1}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %i32) {bin_op=11, ordering=1} : (!llvm.ptr<float>, !llvm.i32) -> !llvm.float
  llvm.return
}

// -----

// CHECK-LABEL: @atomicrmw_mismatched_result
func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr<float>, %f32 : !llvm.float) {
  // expected-error@+1 {{expected LLVM IR result type to match type for operand #1}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %f32) {bin_op=11, ordering=1} : (!llvm.ptr<float>, !llvm.float) -> !llvm.i32
  llvm.return
}

// -----

// CHECK-LABEL: @atomicrmw_expected_float
func @atomicrmw_expected_float(%i32_ptr : !llvm.ptr<i32>, %i32 : !llvm.i32) {
  // expected-error@+1 {{expected LLVM IR floating point type}}
  %0 = llvm.atomicrmw fadd %i32_ptr, %i32 unordered : !llvm.i32
  llvm.return
}

// -----

// CHECK-LABEL: @atomicrmw_unexpected_xchg_type
func @atomicrmw_unexpected_xchg_type(%i1_ptr : !llvm.ptr<i1>, %i1 : !llvm.i1) {
  // expected-error@+1 {{unexpected LLVM IR type for 'xchg' bin_op}}
  %0 = llvm.atomicrmw xchg %i1_ptr, %i1 unordered : !llvm.i1
  llvm.return
}

// -----

// CHECK-LABEL: @atomicrmw_expected_int
func @atomicrmw_expected_int(%f32_ptr : !llvm.ptr<float>, %f32 : !llvm.float) {
  // expected-error@+1 {{expected LLVM IR integer type}}
  %0 = llvm.atomicrmw max %f32_ptr, %f32 unordered : !llvm.float
  llvm.return
}

// -----

// CHECK-LABEL: @cmpxchg_expected_ptr
func @cmpxchg_expected_ptr(%f32_ptr : !llvm.ptr<float>, %f32 : !llvm.float) {
  // expected-error@+1 {{expected LLVM IR pointer type for operand #0}}
  %0 = "llvm.cmpxchg"(%f32, %f32, %f32) {success_ordering=2,failure_ordering=2} : (!llvm.float, !llvm.float, !llvm.float) -> !llvm.struct<(float, i1)>
  llvm.return
}

// -----

// CHECK-LABEL: @cmpxchg_mismatched_operands
func @cmpxchg_mismatched_operands(%i64_ptr : !llvm.ptr<i64>, %i32 : !llvm.i32) {
  // expected-error@+1 {{expected LLVM IR element type for operand #0 to match type for all other operands}}
  %0 = "llvm.cmpxchg"(%i64_ptr, %i32, %i32) {success_ordering=2,failure_ordering=2} : (!llvm.ptr<i64>, !llvm.i32, !llvm.i32) -> !llvm.struct<(i32, i1)>
  llvm.return
}

// -----

// CHECK-LABEL: @cmpxchg_unexpected_type
func @cmpxchg_unexpected_type(%i1_ptr : !llvm.ptr<i1>, %i1 : !llvm.i1) {
  // expected-error@+1 {{unexpected LLVM IR type}}
  %0 = llvm.cmpxchg %i1_ptr, %i1, %i1 monotonic monotonic : !llvm.i1
  llvm.return
}

// -----

// CHECK-LABEL: @cmpxchg_at_least_monotonic_success
func @cmpxchg_at_least_monotonic_success(%i32_ptr : !llvm.ptr<i32>, %i32 : !llvm.i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 unordered monotonic : !llvm.i32
  llvm.return
}

// -----

// CHECK-LABEL: @cmpxchg_at_least_monotonic_failure
func @cmpxchg_at_least_monotonic_failure(%i32_ptr : !llvm.ptr<i32>, %i32 : !llvm.i32) {
  // expected-error@+1 {{ordering must be at least 'monotonic'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 monotonic unordered : !llvm.i32
  llvm.return
}

// -----

// CHECK-LABEL: @cmpxchg_failure_release
func @cmpxchg_failure_release(%i32_ptr : !llvm.ptr<i32>, %i32 : !llvm.i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel release : !llvm.i32
  llvm.return
}

// -----

// CHECK-LABEL: @cmpxchg_failure_acq_rel
func @cmpxchg_failure_acq_rel(%i32_ptr : !llvm.ptr<i32>, %i32 : !llvm.i32) {
  // expected-error@+1 {{failure ordering cannot be 'release' or 'acq_rel'}}
  %0 = llvm.cmpxchg %i32_ptr, %i32, %i32 acq_rel acq_rel : !llvm.i32
  llvm.return
}

// -----

llvm.func @foo(!llvm.i32) -> !llvm.i32
llvm.func @__gxx_personality_v0(...) -> !llvm.i32

llvm.func @bad_landingpad(%arg0: !llvm.ptr<ptr<i8>>) attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(3 : i32) : !llvm.i32
  %1 = llvm.mlir.constant(2 : i32) : !llvm.i32
  %2 = llvm.invoke @foo(%1) to ^bb1 unwind ^bb2 : (!llvm.i32) -> !llvm.i32
^bb1:  // pred: ^bb0
  llvm.return %1 : !llvm.i32
^bb2:  // pred: ^bb0
  // expected-error@+1 {{clause #0 is not a known constant - null, addressof, bitcast}}
  %3 = llvm.landingpad cleanup (catch %1 : !llvm.i32) (catch %arg0 : !llvm.ptr<ptr<i8>>) : !llvm.struct<(ptr<i8>, i32)>
  llvm.return %0 : !llvm.i32
}

// -----

llvm.func @foo(!llvm.i32) -> !llvm.i32
llvm.func @__gxx_personality_v0(...) -> !llvm.i32

llvm.func @caller(%arg0: !llvm.i32) -> !llvm.i32 attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %1 = llvm.alloca %0 x !llvm.ptr<i8> : (!llvm.i32) -> !llvm.ptr<ptr<i8>>
  // expected-note@+1 {{global addresses expected as operand to bitcast used in clauses for landingpad}}
  %2 = llvm.bitcast %1 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  %3 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (!llvm.i32) -> !llvm.i32
^bb1: // pred: ^bb0
  llvm.return %0 : !llvm.i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{constant clauses expected}}
  %5 = llvm.landingpad (catch %2 : !llvm.ptr<i8>) : !llvm.struct<(ptr<i8>, i32)>
  llvm.return %0 : !llvm.i32
}

// -----

llvm.func @foo(!llvm.i32) -> !llvm.i32
llvm.func @__gxx_personality_v0(...) -> !llvm.i32

llvm.func @caller(%arg0: !llvm.i32) -> !llvm.i32 attributes { personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (!llvm.i32) -> !llvm.i32
^bb1: // pred: ^bb0
  llvm.return %0 : !llvm.i32
^bb2: // pred: ^bb0
  // expected-error@+1 {{landingpad instruction expects at least one clause or cleanup attribute}}
  %2 = llvm.landingpad : !llvm.struct<(ptr<i8>, i32)>
  llvm.return %0 : !llvm.i32
}

// -----

llvm.func @foo(!llvm.i32) -> !llvm.i32
llvm.func @__gxx_personality_v0(...) -> !llvm.i32

llvm.func @caller(%arg0: !llvm.i32) -> !llvm.i32 attributes { personality = @__gxx_personality_v0 } {
  %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (!llvm.i32) -> !llvm.i32
^bb1: // pred: ^bb0
  llvm.return %0 : !llvm.i32
^bb2: // pred: ^bb0
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr<i8>, i32)>
  // expected-error@+1 {{'llvm.resume' op expects landingpad value as operand}}
  llvm.resume %0 : !llvm.i32
}

// -----

llvm.func @foo(!llvm.i32) -> !llvm.i32

llvm.func @caller(%arg0: !llvm.i32) -> !llvm.i32 {
  %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %1 = llvm.invoke @foo(%0) to ^bb1 unwind ^bb2 : (!llvm.i32) -> !llvm.i32
^bb1: // pred: ^bb0
  llvm.return %0 : !llvm.i32
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
