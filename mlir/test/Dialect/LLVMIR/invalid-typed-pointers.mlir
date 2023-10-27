// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @alloca_ptr_type_attr_non_opaque_ptr(%sz : i64) {
  // expected-error@below {{unexpected 'elem_type' attribute when non-opaque pointer type is used}}
  "llvm.alloca"(%sz) { elem_type = i32 } : (i64) -> !llvm.ptr<i32>
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

func.func @indirect_callee_arg_mismatch(%arg0 : i32, %callee : !llvm.ptr<func<void(i8)>>) {
  // expected-error@+1 {{'llvm.call' op operand type mismatch for operand 0: 'i32' != 'i8'}}
  "llvm.call"(%callee, %arg0) : (!llvm.ptr<func<void(i8)>>, i32) -> ()
  llvm.return
}

// -----

func.func @indirect_callee_return_mismatch(%callee : !llvm.ptr<func<i8()>>) {
  // expected-error@+1 {{'llvm.call' op result type mismatch: 'i32' != 'i8'}}
  "llvm.call"(%callee) : (!llvm.ptr<func<i8()>>) -> (i32)
  llvm.return
}

// -----

func.func @atomicrmw_mismatched_operands(%f32_ptr : !llvm.ptr<f32>, %i32 : i32) {
  // expected-error@+1 {{expected LLVM IR element type for operand #0 to match type for operand #1}}
  %0 = "llvm.atomicrmw"(%f32_ptr, %i32) {bin_op=11, ordering=1} : (!llvm.ptr<f32>, i32) -> i32
  llvm.return
}

// -----

func.func @cmpxchg_expected_ptr(%f32 : f32) {
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

// expected-error@below {{'llvm.resume' should have a consistent input type inside a function}}
llvm.func @caller(%arg0: i32, %arg1: !llvm.struct<(ptr<i32>, i32)>) -> i32 attributes { personality = @__gxx_personality_v0 } {
  %0 = llvm.invoke @foo(%arg0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1:
  %1 = llvm.invoke @foo(%0) to ^bb3 unwind ^bb4 : (i32) -> i32
^bb2:
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr<i8>, i32)>
  llvm.resume %arg1 : !llvm.struct<(ptr<i32>, i32)>
^bb3:
  llvm.return %1 : i32
^bb4:
  %3 = llvm.landingpad cleanup : !llvm.struct<(ptr<i8>, i32)>
  llvm.resume %3 : !llvm.struct<(ptr<i8>, i32)>
}

// -----

llvm.func @foo(i32) -> i32
llvm.func @__gxx_personality_v0(...) -> i32

// expected-error@below {{'llvm.landingpad' should have a consistent result type inside a function}}
llvm.func @caller(%arg0: i32) -> i32 attributes { personality = @__gxx_personality_v0 } {
  %0 = llvm.invoke @foo(%arg0) to ^bb1 unwind ^bb2 : (i32) -> i32
^bb1:
  %1 = llvm.invoke @foo(%0) to ^bb3 unwind ^bb4 : (i32) -> i32
^bb2:
  %2 = llvm.landingpad cleanup : !llvm.struct<(ptr<i8>, i32)>
  llvm.resume %2 : !llvm.struct<(ptr<i8>, i32)>
^bb3:
  llvm.return %1 : i32
^bb4:
  %3 = llvm.landingpad cleanup : !llvm.struct<(ptr<i32>, i32)>
  llvm.resume %3 : !llvm.struct<(ptr<i32>, i32)>
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

llvm.func @wmmaLoadOp_invalid_mem_space(%arg0: !llvm.ptr<5>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected source pointer in memory space 0, 1, 3}}
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<5>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_AOp(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 8 elements of type 'vector<2xf16>'}}
  %0 = nvvm.wmma.load %arg0, %arg1
  {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_BOp(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 8 elements of type 'vector<2xf16>'}}
 %0 = nvvm.wmma.load %arg0, %arg1
 {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<b>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
 : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// -----

llvm.func @wmmaLoadOp_invalid_COp(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // expected-error@+1 {{'nvvm.wmma.load' op expected destination type is a structure of 4 elements of type 'vector<2xf16>'}}
 %0 = nvvm.wmma.load %arg0, %arg1
   {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<c>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
   : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// -----

llvm.func @wmmaStoreOp_invalid_mem_space(%arg0: !llvm.ptr<5>, %arg1: i32,
                            %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                            %arg4: vector<2 x f16>, %arg5: vector<2 xf16>) {
  // expected-error@+1 {{'nvvm.wmma.store' op expected operands to be a source pointer in memory space 0, 1, 3}}
  nvvm.wmma.store %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : !llvm.ptr<5>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
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

func.func @cp_async(%arg0: !llvm.ptr<i8, 3>, %arg1: !llvm.ptr<i8, 1>) {
  // expected-error @below {{expected byte size to be either 4, 8 or 16.}}
  nvvm.cp.async.shared.global %arg0, %arg1, 32, cache = ca : !llvm.ptr<i8, 3>, !llvm.ptr<i8, 1>
  return
}

// -----

func.func @cp_async(%arg0: !llvm.ptr<i8, 3>, %arg1: !llvm.ptr<i8, 1>) {
  // expected-error @below {{CG cache modifier is only support for 16 bytes copy.}}
  nvvm.cp.async.shared.global %arg0, %arg1, 8, cache = cg : !llvm.ptr<i8, 3>, !llvm.ptr<i8, 1>
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
