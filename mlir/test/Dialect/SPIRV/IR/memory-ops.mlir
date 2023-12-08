// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.AccessChain
//===----------------------------------------------------------------------===//

func.func @access_chain_struct() -> () {
  %0 = spirv.Constant 1: i32
  %1 = spirv.Variable : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>
  // CHECK: spirv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4 x f32>)>, Function>
  %2 = spirv.AccessChain %1[%0, %0] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>, i32, i32
  return
}

func.func @access_chain_1D_array(%arg0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4xf32>, Function>
  // CHECK: spirv.AccessChain {{.*}}[{{.*}}] : !spirv.ptr<!spirv.array<4 x f32>, Function>
  %1 = spirv.AccessChain %0[%arg0] : !spirv.ptr<!spirv.array<4xf32>, Function>, i32
  return
}

func.func @access_chain_2D_array_1(%arg0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // CHECK: spirv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32>>, Function>
  %1 = spirv.AccessChain %0[%arg0, %arg0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32, i32
  %2 = spirv.Load "Function" %1 ["Volatile"] : f32
  return
}

func.func @access_chain_2D_array_2(%arg0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // CHECK: spirv.AccessChain {{.*}}[{{.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32>>, Function>
  %1 = spirv.AccessChain %0[%arg0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32
  %2 = spirv.Load "Function" %1 ["Volatile"] : !spirv.array<4xf32>
  return
}

func.func @access_chain_rtarray(%arg0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.rtarray<f32>, Function>
  // CHECK: spirv.AccessChain {{.*}}[{{.*}}] : !spirv.ptr<!spirv.rtarray<f32>, Function>
  %1 = spirv.AccessChain %0[%arg0] : !spirv.ptr<!spirv.rtarray<f32>, Function>, i32
  %2 = spirv.Load "Function" %1 ["Volatile"] : f32
  return
}

// -----

func.func @access_chain_non_composite() -> () {
  %0 = spirv.Constant 1: i32
  %1 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 0}}
  %2 = spirv.AccessChain %1[%0] : !spirv.ptr<f32, Function>, i32
  return
}

// -----

func.func @access_chain_no_indices(%index0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // expected-error @+1 {{expected at least one index}}
  %1 = spirv.AccessChain %0[] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32
  return
}

// -----

func.func @access_chain_missing_comma(%index0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // expected-error @+1 {{expected ','}}
  %1 = spirv.AccessChain %0[%index0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function> i32
  return
}

// -----

func.func @access_chain_invalid_indices_types_count(%index0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // expected-error @+1 {{'spirv.AccessChain' op indices types' count must be equal to indices info count}}
  %1 = spirv.AccessChain %0[%index0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32, i32
  return
}

// -----

func.func @access_chain_missing_indices_type(%index0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // expected-error @+1 {{'spirv.AccessChain' op indices types' count must be equal to indices info count}}
  %1 = spirv.AccessChain %0[%index0, %index0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32
  return
}

// -----

func.func @access_chain_invalid_type(%index0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  %1 = spirv.Load "Function" %0 ["Volatile"] : !spirv.array<4x!spirv.array<4xf32>>
  // expected-error @+1 {{expected a pointer to composite type, but provided '!spirv.array<4 x !spirv.array<4 x f32>>'}}
  %2 = spirv.AccessChain %1[%index0] : !spirv.array<4x!spirv.array<4xf32>>, i32
  return
}

// -----

func.func @access_chain_invalid_index_1(%index0 : i32) -> () {
   %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // expected-error @+1 {{expected SSA operand}}
  %1 = spirv.AccessChain %0[%index, 4] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32, i32
  return
}

// -----

func.func @access_chain_invalid_index_2(%index0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>
  // expected-error @+1 {{index must be an integer spirv.Constant to access element of spirv.struct}}
  %1 = spirv.AccessChain %0[%index0, %index0] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>, i32, i32
  return
}

// -----

func.func @access_chain_invalid_constant_type_1() -> () {
  %0 = arith.constant 1: i32
  %1 = spirv.Variable : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>
  // expected-error @+1 {{index must be an integer spirv.Constant to access element of spirv.struct, but provided arith.constant}}
  %2 = spirv.AccessChain %1[%0, %0] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>, i32, i32
  return
}

// -----

func.func @access_chain_out_of_bounds() -> () {
  %index0 = "spirv.Constant"() { value = 12: i32} : () -> i32
  %0 = spirv.Variable : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>
  // expected-error @+1 {{'spirv.AccessChain' op index 12 out of bounds for '!spirv.struct<(f32, !spirv.array<4 x f32>)>'}}
  %1 = spirv.AccessChain %0[%index0, %index0] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Function>, i32, i32
  return
}

// -----

func.func @access_chain_invalid_accessing_type(%index0 : i32) -> () {
  %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 0}}
  %1 = spirv.AccessChain %0[%index, %index0, %index0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32, i32, i32
  return

// -----

//===----------------------------------------------------------------------===//
// spirv.LoadOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @simple_load
func.func @simple_load() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load "Function" %{{.*}} : f32
  %1 = spirv.Load "Function" %0 : f32
  return
}

// CHECK-LABEL: @load_none_access
func.func @load_none_access() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load "Function" %{{.*}} ["None"] : f32
  %1 = spirv.Load "Function" %0 ["None"] : f32
  return
}

// CHECK-LABEL: @volatile_load
func.func @volatile_load() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load "Function" %{{.*}} ["Volatile"] : f32
  %1 = spirv.Load "Function" %0 ["Volatile"] : f32
  return
}

// CHECK-LABEL: @aligned_load
func.func @aligned_load() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load "Function" %{{.*}} ["Aligned", 4] : f32
  %1 = spirv.Load "Function" %0 ["Aligned", 4] : f32
  return
}

// CHECK-LABEL: @volatile_aligned_load
func.func @volatile_aligned_load() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load "Function" %{{.*}} ["Volatile|Aligned", 4] : f32
  %1 = spirv.Load "Function" %0 ["Volatile|Aligned", 4] : f32
  return
}

// -----

// CHECK-LABEL: load_none_access
func.func @load_none_access() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load
  // CHECK-SAME: ["None"]
  %1 = "spirv.Load"(%0) {memory_access = #spirv.memory_access<None>} : (!spirv.ptr<f32, Function>) -> (f32)
  return
}

// CHECK-LABEL: volatile_load
func.func @volatile_load() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load
  // CHECK-SAME: ["Volatile"]
  %1 = "spirv.Load"(%0) {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<f32, Function>) -> (f32)
  return
}

// CHECK-LABEL: aligned_load
func.func @aligned_load() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load
  // CHECK-SAME: ["Aligned", 4]
  %1 = "spirv.Load"(%0) {memory_access = #spirv.memory_access<Aligned>, alignment = 4 : i32} : (!spirv.ptr<f32, Function>) -> (f32)
  return
}

// CHECK-LABEL: volatile_aligned_load
func.func @volatile_aligned_load() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Load
  // CHECK-SAME: ["Volatile|Aligned", 4]
  %1 = "spirv.Load"(%0) {memory_access = #spirv.memory_access<Volatile|Aligned>, alignment = 4 : i32} : (!spirv.ptr<f32, Function>) -> (f32)
  return
}

// -----

func.func @simple_load_missing_storageclass() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected attribute value}}
  %1 = spirv.Load %0 : f32
  return
}

// -----

func.func @simple_load_missing_operand() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected SSA operand}}
  %1 = spirv.Load "Function" : f32
  return
}

// -----

func.func @simple_load_missing_rettype() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ':'}}
  %1 = spirv.Load "Function" %0
  return
}

// -----

func.func @volatile_load_missing_lbrace() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ':'}}
  %1 = spirv.Load "Function" %0 "Volatile"] : f32
  return
}

// -----

func.func @volatile_load_missing_rbrace() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spirv.Load "Function" %0 ["Volatile"} : f32
  return
}

// -----

func.func @aligned_load_missing_alignment() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  %1 = spirv.Load "Function" %0 ["Aligned"] : f32
  return
}

// -----

func.func @aligned_load_missing_comma() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  %1 = spirv.Load "Function" %0 ["Aligned" 4] : f32
  return
}

// -----

func.func @load_incorrect_attributes() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spirv.Load "Function" %0 ["Volatile", 4] : f32
  return
}

// -----

func.func @load_unknown_memory_access() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spirv.Load' invalid memory_access attribute specification: "Something"}}
  %1 = spirv.Load "Function" %0 ["Something"] : f32
  return
}

// -----

func.func @load_unknown_memory_access() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spirv.Load' invalid memory_access attribute specification: "Volatile|Something"}}
  %1 = spirv.Load "Function" %0 ["Volatile|Something"] : f32
  return
}

// -----

func.func @load_unknown_memory_access() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{failed to satisfy constraint: valid SPIR-V MemoryAccess}}
  %1 = "spirv.Load"(%0) {memory_access = 0x80000000 : i32} : (!spirv.ptr<f32, Function>) -> (f32)
  return
}

// -----

func.func @aligned_load_incorrect_attributes() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spirv.Load "Function" %0 ["Aligned", 4, 23] : f32
  return
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, Arrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant>
  // CHECK-LABEL: @simple_load
  spirv.func @simple_load() -> () "None" {
    // CHECK: spirv.Load "Input" {{%.*}} : f32
    %0 = spirv.mlir.addressof @var0 : !spirv.ptr<f32, Input>
    %1 = spirv.Load "Input" %0 : f32
    %2 = spirv.mlir.addressof @var1 : !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, Arrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant>
    // CHECK: spirv.Load "UniformConstant" {{%.*}} : !spirv.sampled_image
    %3 = spirv.Load "UniformConstant" %2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, Arrayed, SingleSampled, NeedSampler, Unknown>>
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.StoreOp
//===----------------------------------------------------------------------===//

func.func @simple_store(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Store  "Function" %0, %arg0 : f32
  spirv.Store  "Function" %0, %arg0 : f32
  return
}

// CHECK-LABEL: @volatile_store
func.func @volatile_store(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Store  "Function" %0, %arg0 ["Volatile"] : f32
  spirv.Store  "Function" %0, %arg0 ["Volatile"] : f32
  return
}

// CHECK-LABEL: @aligned_store
func.func @aligned_store(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.Store  "Function" %0, %arg0 ["Aligned", 4] : f32
  spirv.Store  "Function" %0, %arg0 ["Aligned", 4] : f32
  return
}

// -----

func.func @simple_store_missing_ptr_type(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected attribute value}}
  spirv.Store  %0, %arg0 : f32
  return
}

// -----

func.func @simple_store_missing_operand(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected operand}}
  spirv.Store  "Function" , %arg0 : f32
  return
}

// -----

func.func @simple_store_missing_operand(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spirv.Store' expected 2 operands}}
  spirv.Store  "Function" %0 : f32
  return
}

// -----

func.func @volatile_store_missing_lbrace(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ':'}}
  spirv.Store  "Function" %0, %arg0 "Volatile"] : f32
  return
}

// -----

func.func @volatile_store_missing_rbrace(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spirv.Store "Function" %0, %arg0 ["Volatile"} : f32
  return
}

// -----

func.func @aligned_store_missing_alignment(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  spirv.Store  "Function" %0, %arg0 ["Aligned"] : f32
  return
}

// -----

func.func @aligned_store_missing_comma(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  spirv.Store  "Function" %0, %arg0 ["Aligned" 4] : f32
  return
}

// -----

func.func @load_incorrect_attributes(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spirv.Store  "Function" %0, %arg0 ["Volatile", 4] : f32
  return
}

// -----

func.func @aligned_store_incorrect_attributes(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spirv.Store  "Function" %0, %arg0 ["Aligned", 4, 23] : f32
  return
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input>
  spirv.func @simple_store(%arg0 : f32) -> () "None" {
    %0 = spirv.mlir.addressof @var0 : !spirv.ptr<f32, Input>
    // CHECK: spirv.Store  "Input" {{%.*}}, {{%.*}} : f32
    spirv.Store  "Input" %0, %arg0 : f32
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Variable
//===----------------------------------------------------------------------===//

func.func @variable(%arg0: f32) -> () {
  // CHECK: spirv.Variable : !spirv.ptr<f32, Function>
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  return
}

// -----

func.func @variable_init_normal_constant() -> () {
  // CHECK: %[[cst:.*]] = spirv.Constant
  %0 = spirv.Constant 4.0 : f32
  // CHECK: spirv.Variable init(%[[cst]]) : !spirv.ptr<f32, Function>
  %1 = spirv.Variable init(%0) : !spirv.ptr<f32, Function>
  return
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @global : !spirv.ptr<f32, Workgroup>
  spirv.func @variable_init_global_variable() -> () "None" {
    %0 = spirv.mlir.addressof @global : !spirv.ptr<f32, Workgroup>
    // CHECK: spirv.Variable init({{.*}}) : !spirv.ptr<!spirv.ptr<f32, Workgroup>, Function>
    %1 = spirv.Variable init(%0) : !spirv.ptr<!spirv.ptr<f32, Workgroup>, Function>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc = 42 : i32
  // CHECK-LABEL: @variable_init_spec_constant
  spirv.func @variable_init_spec_constant() -> () "None" {
    %0 = spirv.mlir.referenceof @sc : i32
    // CHECK: spirv.Variable init(%0) : !spirv.ptr<i32, Function>
    %1 = spirv.Variable init(%0) : !spirv.ptr<i32, Function>
    spirv.Return
  }
}

// -----

func.func @variable_ptr_physical_buffer() -> () {
  %0 = spirv.Variable {aliased_pointer} :
    !spirv.ptr<!spirv.ptr<f32, PhysicalStorageBuffer>, Function>
  %1 = spirv.Variable {restrict_pointer} :
    !spirv.ptr<!spirv.ptr<f32, PhysicalStorageBuffer>, Function>
  return
}

// -----

func.func @variable_ptr_physical_buffer_no_decoration() -> () {
  // expected-error @+1 {{must be decorated either 'AliasedPointer' or 'RestrictPointer'}}
  %0 = spirv.Variable : !spirv.ptr<!spirv.ptr<f32, PhysicalStorageBuffer>, Function>
  return
}

// -----

func.func @variable_ptr_physical_buffer_two_alias_decorations() -> () {
  // expected-error @+1 {{must have exactly one aliasing decoration}}
  %0 = spirv.Variable {aliased_pointer, restrict_pointer} :
    !spirv.ptr<!spirv.ptr<f32, PhysicalStorageBuffer>, Function>
  return
}

// -----

func.func @variable_ptr_array_physical_buffer() -> () {
  %0 = spirv.Variable {aliased_pointer} :
    !spirv.ptr<!spirv.array<4x!spirv.ptr<f32, PhysicalStorageBuffer>>, Function>
  %1 = spirv.Variable {restrict_pointer} :
    !spirv.ptr<!spirv.array<4x!spirv.ptr<f32, PhysicalStorageBuffer>>, Function>
  return
}

// -----

func.func @variable_ptr_array_physical_buffer_no_decoration() -> () {
  // expected-error @+1 {{must be decorated either 'AliasedPointer' or 'RestrictPointer'}}
  %0 = spirv.Variable :
    !spirv.ptr<!spirv.array<4x!spirv.ptr<f32, PhysicalStorageBuffer>>, Function>
  return
}

// -----

func.func @variable_ptr_array_physical_buffer_two_alias_decorations() -> () {
  // expected-error @+1 {{must have exactly one aliasing decoration}}
  %0 = spirv.Variable {aliased_pointer, restrict_pointer} :
    !spirv.ptr<!spirv.array<4x!spirv.ptr<f32, PhysicalStorageBuffer>>, Function>
  return
}

// -----

func.func @variable_bind() -> () {
  // expected-error @+1 {{cannot have 'descriptor_set' attribute (only allowed in spirv.GlobalVariable)}}
  %0 = spirv.Variable bind(1, 2) : !spirv.ptr<f32, Function>
  return
}

// -----

func.func @variable_init_bind() -> () {
  %0 = spirv.Constant 4.0 : f32
  // expected-error @+1 {{cannot have 'binding' attribute (only allowed in spirv.GlobalVariable)}}
  %1 = spirv.Variable init(%0) {binding = 5 : i32} : !spirv.ptr<f32, Function>
  return
}

// -----

func.func @variable_builtin() -> () {
  // expected-error @+1 {{cannot have 'built_in' attribute (only allowed in spirv.GlobalVariable)}}
  %1 = spirv.Variable built_in("GlobalInvocationID") : !spirv.ptr<vector<3xi32>, Function>
  return
}

// -----

func.func @expect_ptr_result_type(%arg0: f32) -> () {
  // expected-error @+1 {{expected spirv.ptr type}}
  %0 = spirv.Variable : f32
  return
}

// -----

func.func @variable_init(%arg0: f32) -> () {
  // expected-error @+1 {{op initializer must be the result of a constant or spirv.GlobalVariable op}}
  %0 = spirv.Variable init(%arg0) : !spirv.ptr<f32, Function>
  return
}

// -----

func.func @cannot_be_generic_storage_class(%arg0: f32) -> () {
  // expected-error @+1 {{op can only be used to model function-level variables. Use spirv.GlobalVariable for module-level variables}}
  %0 = spirv.Variable : !spirv.ptr<f32, Generic>
  return
}

// -----

func.func @copy_memory_incompatible_ptrs() {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  %1 = spirv.Variable : !spirv.ptr<i32, Function>
  // expected-error @+1 {{both operands must be pointers to the same type}}
  "spirv.CopyMemory"(%0, %1) {} : (!spirv.ptr<f32, Function>, !spirv.ptr<i32, Function>) -> ()
  spirv.Return
}

// -----

func.func @copy_memory_invalid_maa() {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  %1 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{missing alignment value}}
  "spirv.CopyMemory"(%0, %1) {memory_access=#spirv.memory_access<Aligned>} : (!spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()
  spirv.Return
}

// -----

func.func @copy_memory_invalid_source_maa() {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  %1 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{invalid alignment specification with non-aligned memory access specification}}
  "spirv.CopyMemory"(%0, %1) {source_memory_access=#spirv.memory_access<Volatile>, memory_access=#spirv.memory_access<Aligned>, source_alignment=8 : i32, alignment=4 : i32} : (!spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()
  spirv.Return
}

// -----

func.func @copy_memory_invalid_source_maa2() {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  %1 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error @+1 {{missing alignment value}}
  "spirv.CopyMemory"(%0, %1) {source_memory_access=#spirv.memory_access<Aligned>, memory_access=#spirv.memory_access<Aligned>, alignment=4 : i32} : (!spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()
  spirv.Return
}

// -----

func.func @copy_memory_print_maa() {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  %1 = spirv.Variable : !spirv.ptr<f32, Function>

  // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Volatile"] : f32
  "spirv.CopyMemory"(%0, %1) {memory_access=#spirv.memory_access<Volatile>} : (!spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()

  // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 4] : f32
  "spirv.CopyMemory"(%0, %1) {memory_access=#spirv.memory_access<Aligned>, alignment=4 : i32} : (!spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()

  // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 4], ["Volatile"] : f32
  "spirv.CopyMemory"(%0, %1) {source_memory_access=#spirv.memory_access<Volatile>, memory_access=#spirv.memory_access<Aligned>, alignment=4 : i32} : (!spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()

  // CHECK: spirv.CopyMemory "Function" %{{.*}}, "Function" %{{.*}} ["Aligned", 4], ["Aligned", 8] : f32
  "spirv.CopyMemory"(%0, %1) {source_memory_access=#spirv.memory_access<Aligned>, memory_access=#spirv.memory_access<Aligned>, source_alignment=8 : i32, alignment=4 : i32} : (!spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()

  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.PtrAccessChain
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func @ptr_access_chain1(
// CHECK-SAME:    %[[ARG0:.*]]: !spirv.ptr<f32, CrossWorkgroup>,
// CHECK-SAME:    %[[ARG1:.*]]: i64)
// CHECK: spirv.PtrAccessChain %[[ARG0]][%[[ARG1]]] : !spirv.ptr<f32, CrossWorkgroup>, i64
func.func @ptr_access_chain1(%arg0: !spirv.ptr<f32, CrossWorkgroup>, %arg1 : i64) -> () {
  %0 = spirv.PtrAccessChain %arg0[%arg1] : !spirv.ptr<f32, CrossWorkgroup>, i64
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.InBoundsPtrAccessChain
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func @inbounds_ptr_access_chain1(
// CHECK-SAME:    %[[ARG0:.*]]: !spirv.ptr<f32, CrossWorkgroup>,
// CHECK-SAME:    %[[ARG1:.*]]: i64)
// CHECK: spirv.InBoundsPtrAccessChain %[[ARG0]][%[[ARG1]]] : !spirv.ptr<f32, CrossWorkgroup>, i64
func.func @inbounds_ptr_access_chain1(%arg0: !spirv.ptr<f32, CrossWorkgroup>, %arg1 : i64) -> () {
  %0 = spirv.InBoundsPtrAccessChain %arg0[%arg1] : !spirv.ptr<f32, CrossWorkgroup>, i64
  return
}
