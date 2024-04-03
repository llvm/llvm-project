// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.mlir.addressof
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
  spirv.func @access_chain() -> () "None" {
    %0 = spirv.Constant 1: i32
    // CHECK: [[VAR1:%.*]] = spirv.mlir.addressof @var1 : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4 x f32>)>, Input>
    // CHECK-NEXT: spirv.AccessChain [[VAR1]][{{.*}}, {{.*}}] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4 x f32>)>, Input>
    %1 = spirv.mlir.addressof @var1 : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
    %2 = spirv.AccessChain %1[%0, %0] : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>, i32, i32
    spirv.Return
  }
}

// -----

// Allow taking address of global variables in other module-like ops
spirv.GlobalVariable @var : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
func.func @addressof() -> () {
  // CHECK: spirv.mlir.addressof @var
  %1 = spirv.mlir.addressof @var : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
  return
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
  spirv.func @foo() -> () "None" {
    // expected-error @+1 {{expected spirv.GlobalVariable symbol}}
    %0 = spirv.mlir.addressof @var2 : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.struct<(f32, !spirv.array<4xf32>)>, Input>
  spirv.func @foo() -> () "None" {
    // expected-error @+1 {{result type mismatch with the referenced global variable's type}}
    %0 = spirv.mlir.addressof @var1 : !spirv.ptr<f32, Input>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Constant
//===----------------------------------------------------------------------===//

func.func @const() -> () {
  // CHECK: spirv.Constant true
  // CHECK: spirv.Constant 42 : i32
  // CHECK: spirv.Constant 5.000000e-01 : f32
  // CHECK: spirv.Constant dense<[2, 3]> : vector<2xi32>
  // CHECK: spirv.Constant [dense<3.000000e+00> : vector<2xf32>] : !spirv.array<1 x vector<2xf32>>
  // CHECK: spirv.Constant dense<1> : tensor<2x3xi32> : !spirv.array<2 x !spirv.array<3 x i32>>
  // CHECK: spirv.Constant dense<1.000000e+00> : tensor<2x3xf32> : !spirv.array<2 x !spirv.array<3 x f32>>
  // CHECK: spirv.Constant dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32> : !spirv.array<2 x !spirv.array<3 x i32>>
  // CHECK: spirv.Constant dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32> : !spirv.array<2 x !spirv.array<3 x f32>>

  %0 = spirv.Constant true
  %1 = spirv.Constant 42 : i32
  %2 = spirv.Constant 0.5 : f32
  %3 = spirv.Constant dense<[2, 3]> : vector<2xi32>
  %4 = spirv.Constant [dense<3.0> : vector<2xf32>] : !spirv.array<1xvector<2xf32>>
  %5 = spirv.Constant dense<1> : tensor<2x3xi32> : !spirv.array<2 x !spirv.array<3 x i32>>
  %6 = spirv.Constant dense<1.0> : tensor<2x3xf32> : !spirv.array<2 x !spirv.array<3 x f32>>
  %7 = spirv.Constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32> : !spirv.array<2 x !spirv.array<3 x i32>>
  %8 = spirv.Constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32> : !spirv.array<2 x !spirv.array<3 x f32>>
  %9 = spirv.Constant [[dense<3.0> : vector<2xf32>]] : !spirv.array<1 x !spirv.array<1xvector<2xf32>>>
  return
}

// -----

func.func @unaccepted_std_attr() -> () {
  // expected-error @+1 {{cannot have attribute: unit}}
  %0 = spirv.Constant unit : none
  return
}

// -----

func.func @array_constant() -> () {
  // expected-error @+1 {{result or element type ('vector<2xf32>') does not match value type ('vector<2xi32>')}}
  %0 = spirv.Constant [dense<3.0> : vector<2xf32>, dense<4> : vector<2xi32>] : !spirv.array<2xvector<2xf32>>
  return
}

// -----

func.func @array_constant() -> () {
  // expected-error @+1 {{must have spirv.array result type for array value}}
  %0 = spirv.Constant [dense<3.0> : vector<2xf32>] : !spirv.rtarray<vector<2xf32>>
  return
}

// -----

func.func @non_nested_array_constant() -> () {
  // expected-error @+1 {{only support nested array result type}}
  %0 = spirv.Constant dense<3.0> : tensor<2x2xf32> : !spirv.array<2xvector<2xf32>>
  return
}

// -----

func.func @value_result_type_mismatch() -> () {
  // expected-error @+1 {{result or element type ('vector<4xi32>') does not match value type ('tensor<4xi32>')}}
  %0 = "spirv.Constant"() {value = dense<0> : tensor<4xi32>} : () -> (vector<4xi32>)
}

// -----

func.func @value_result_type_mismatch() -> () {
  // expected-error @+1 {{result element type ('i32') does not match value element type ('f32')}}
  %0 = spirv.Constant dense<1.0> : tensor<2x3xf32> : !spirv.array<2 x !spirv.array<3 x i32>>
}

// -----

func.func @value_result_num_elements_mismatch() -> () {
  // expected-error @+1 {{result number of elements (6) does not match value number of elements (4)}}
  %0 = spirv.Constant dense<1.0> : tensor<2x2xf32> : !spirv.array<2 x !spirv.array<3 x f32>>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.EntryPoint
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   // CHECK: spirv.EntryPoint "GLCompute" @do_nothing
   spirv.EntryPoint "GLCompute" @do_nothing
}

spirv.module Logical GLSL450 {
   spirv.GlobalVariable @var2 : !spirv.ptr<f32, Input>
   spirv.GlobalVariable @var3 : !spirv.ptr<f32, Output>
   spirv.func @do_something(%arg0 : !spirv.ptr<f32, Input>, %arg1 : !spirv.ptr<f32, Output>) -> () "None" {
     %1 = spirv.Load "Input" %arg0 : f32
     spirv.Store "Output" %arg1, %1 : f32
     spirv.Return
   }
   // CHECK: spirv.EntryPoint "GLCompute" @do_something, @var2, @var3
   spirv.EntryPoint "GLCompute" @do_something, @var2, @var3
}

// -----

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   // expected-error @+1 {{invalid kind of attribute specified}}
   spirv.EntryPoint "GLCompute" "do_nothing"
}

// -----

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   // expected-error @+1 {{function 'do_something' not found in 'spirv.module'}}
   spirv.EntryPoint "GLCompute" @do_something
}

/// TODO: Add a test that verifies an error is thrown
/// when interface entries of EntryPointOp are not
/// spirv.Variables. There is currently no other op that has a spirv.ptr
/// return type

// -----

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     // expected-error @+1 {{op must appear in a module-like op's block}}
     spirv.EntryPoint "GLCompute" @do_something
   }
}

// -----

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   spirv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{duplicate of a previous EntryPointOp}}
   spirv.EntryPoint "GLCompute" @do_nothing
}

// -----

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   spirv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{'spirv.EntryPoint' invalid execution_model attribute specification: "ContractionOff"}}
   spirv.EntryPoint "ContractionOff" @do_nothing
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ExecutionMode
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   spirv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spirv.ExecutionMode {{@.*}} "ContractionOff"
   spirv.ExecutionMode @do_nothing "ContractionOff"
}

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   spirv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spirv.ExecutionMode {{@.*}} "LocalSizeHint", 3, 4, 5
   spirv.ExecutionMode @do_nothing "LocalSizeHint", 3, 4, 5
}

// -----

spirv.module Logical GLSL450 {
   spirv.func @do_nothing() -> () "None" {
     spirv.Return
   }
   spirv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{custom op 'spirv.ExecutionMode' invalid execution_mode attribute specification: "GLCompute"}}
   spirv.ExecutionMode @do_nothing "GLCompute", 3, 4, 5
}

// -----

//===----------------------------------------------------------------------===//
// spirv.func
//===----------------------------------------------------------------------===//

// CHECK: spirv.func @foo() "None"
spirv.func @foo() "None"

// CHECK: spirv.func @bar(%{{.+}}: i32) -> i32 "Inline|Pure" {
spirv.func @bar(%arg: i32) -> (i32) "Inline|Pure" {
  // CHECK-NEXT: spirv.
  spirv.ReturnValue %arg: i32
// CHECK-NEXT: }
}

// CHECK: spirv.func @baz(%{{.+}}: i32) "DontInline" attributes {additional_stuff = 64 : i64}
spirv.func @baz(%arg: i32) "DontInline" attributes {
  additional_stuff = 64
} { spirv.Return }

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
    // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = "outside.func", linkage_type = <Import>>
    spirv.func @outside.func.with.linkage(%arg0 : i8) -> () "Pure" attributes {
      linkage_attributes=#spirv.linkage_attributes<
        linkage_name="outside.func",
        linkage_type=<Import>
      >
    }
    spirv.func @inside.func() -> () "Pure" attributes {} {spirv.Return}
}
// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> { 
  // expected-error @+1 {{'spirv.module' cannot contain external functions without 'Import' linkage_attributes (LinkageAttributes)}}
  spirv.func @outside.func.without.linkage(%arg0 : i8) -> () "Pure"
  spirv.func @inside.func() -> () "Pure" attributes {} {spirv.Return}
}

// -----

// expected-error @+1 {{expected function_control attribute specified as string}}
spirv.func @missing_function_control() { spirv.Return }

// -----

// expected-error @+1 {{cannot have more than one result}}
spirv.func @cannot_have_more_than_one_result(%arg: i32) -> (i32, i32) "None"

// -----

// expected-error @+1 {{expected SSA identifier}}
spirv.func @cannot_have_variadic_arguments(%arg: i32, ...) "None"

// -----

// Nested function
spirv.module Logical GLSL450 {
  spirv.func @outer_func() -> () "None" {
    // expected-error @+1 {{must appear in a module-like op's block}}
    spirv.func @inner_func() -> () "None" {
      spirv.Return
    }
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GlobalVariable
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input>
}

// TODO: Fix test case after initialization with normal constant is addressed
// spirv.module Logical GLSL450 {
//   %0 = spirv.Constant 4.0 : f32
//   // CHECK1: spirv.Variable init(%0) : !spirv.ptr<f32, Private>
//   spirv.GlobalVariable @var1 init(%0) : !spirv.ptr<f32, Private>
// }

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc = 4.0 : f32
  // CHECK: spirv.GlobalVariable @var initializer(@sc) : !spirv.ptr<f32, Private>
  spirv.GlobalVariable @var initializer(@sc) : !spirv.ptr<f32, Private>
}

// -----

// Allow initializers coming from other module-like ops
spirv.SpecConstant @sc = 4.0 : f32
// CHECK: spirv.GlobalVariable @var initializer(@sc)
spirv.GlobalVariable @var initializer(@sc) : !spirv.ptr<f32, Private>


// -----
// Allow SpecConstantComposite as initializer
  spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1 : i8
  spirv.SpecConstant @sc2 = 2 : i8
  spirv.SpecConstant @sc3 = 3 : i8
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.array<3 x i8>

  // CHECK: spirv.GlobalVariable @var initializer(@scc) : !spirv.ptr<!spirv.array<3 x i8>, Private>
  spirv.GlobalVariable @var initializer(@scc) : !spirv.ptr<!spirv.array<3 x i8>, Private>
}

// -----

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @var0 bind(1, 2) : !spirv.ptr<f32, Uniform>
  spirv.GlobalVariable @var0 bind(1, 2) : !spirv.ptr<f32, Uniform>
}

// TODO: Fix test case after initialization with constant is addressed
// spirv.module Logical GLSL450 {
//   %0 = spirv.Constant 4.0 : f32
//   // CHECK1: spirv.GlobalVariable @var1 initializer(%0) {binding = 5 : i32} : !spirv.ptr<f32, Private>
//   spirv.GlobalVariable @var1 initializer(%0) {binding = 5 : i32} : !spirv.ptr<f32, Private>
// }

// -----

spirv.module Logical GLSL450 {
  // CHECK: spirv.GlobalVariable @var1 built_in("GlobalInvocationID") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @var1 built_in("GlobalInvocationID") : !spirv.ptr<vector<3xi32>, Input>
  // CHECK: spirv.GlobalVariable @var2 built_in("GlobalInvocationID") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @var2 {built_in = "GlobalInvocationID"} : !spirv.ptr<vector<3xi32>, Input>
}

// -----

// Allow in other module-like ops
module {
  // CHECK: spirv.GlobalVariable
  spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input>
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
  // CHECK: linkage_attributes = #spirv.linkage_attributes<linkage_name = "outSideGlobalVar1", linkage_type = <Import>>
  spirv.GlobalVariable @var1 {
    linkage_attributes=#spirv.linkage_attributes<
      linkage_name="outSideGlobalVar1", 
      linkage_type=<Import>
    >
  } : !spirv.ptr<f32, Private>
}


// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{expected spirv.ptr type}}
  spirv.GlobalVariable @var0 : f32
}

// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{result must be of a !spv.ptr type}}
  "spirv.GlobalVariable"() {sym_name = "var0", type = none} : () -> ()
}

// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{op initializer must be result of a spirv.SpecConstant or spirv.GlobalVariable or spirv.SpecConstantCompositeOp op}}
  spirv.GlobalVariable @var0 initializer(@var1) : !spirv.ptr<f32, Private>
}

// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{storage class cannot be 'Generic'}}
  spirv.GlobalVariable @var0 : !spirv.ptr<f32, Generic>
}

// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{storage class cannot be 'Function'}}
  spirv.GlobalVariable @var0 : !spirv.ptr<f32, Function>
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @foo() "None" {
    // expected-error @+1 {{op must appear in a module-like op's block}}
    spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input>
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.module
//===----------------------------------------------------------------------===//

// Module without capability and extension
// CHECK: spirv.module Logical GLSL450
spirv.module Logical GLSL450 { }

// Module with a name
// CHECK: spirv.module @{{.*}} Logical GLSL450
spirv.module @name Logical GLSL450 { }

// Module with (version, capabilities, extensions) triple
// CHECK: spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]>
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]> { }

// Module with additional attributes
// CHECK: spirv.module Logical GLSL450 attributes {foo = "bar"}
spirv.module Logical GLSL450 attributes {foo = "bar"} { }

// Module with VCE triple and additional attributes
// CHECK: spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]> attributes {foo = "bar"}
spirv.module Logical GLSL450
  requires #spirv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]>
  attributes {foo = "bar"} { }

// Module with function
// CHECK: spirv.module
spirv.module Logical GLSL450 {
  spirv.func @do_nothing() -> () "None" {
    spirv.Return
  }
}

// -----

// Missing addressing model
// expected-error@+1 {{'spirv.module' expected valid keyword}}
spirv.module { }

// -----

// Wrong addressing model
// expected-error@+1 {{'spirv.module' invalid addressing_model attribute specification: Physical}}
spirv.module Physical { }

// -----

// Missing memory model
// expected-error@+1 {{'spirv.module' expected valid keyword}}
spirv.module Logical { }

// -----

// Wrong memory model
// expected-error@+1 {{'spirv.module' invalid memory_model attribute specification: Bla}}
spirv.module Logical Bla { }

// -----

// Module with multiple blocks
// expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
spirv.module Logical GLSL450 {
^first:
  spirv.Return
^second:
  spirv.Return
}

// -----

// Use non SPIR-V op inside module
spirv.module Logical GLSL450 {
  // expected-error @+1 {{'spirv.module' can only contain spirv.* ops}}
  "dialect.op"() : () -> ()
}

// -----

// Use non SPIR-V op inside function
spirv.module Logical GLSL450 {
  spirv.func @do_nothing() -> () "None" {
    // expected-error @+1 {{functions in 'spirv.module' can only contain spirv.* ops}}
    "dialect.op"() : () -> ()
  }
}

// -----

// Use external function
spirv.module Logical GLSL450 {
  // expected-error @+1 {{'spirv.module' cannot contain external functions}}
  spirv.func @extern() -> () "None"
}

// -----

//===----------------------------------------------------------------------===//
// spirv.mlir.referenceof
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = false
  spirv.SpecConstant @sc2 = 42 : i64
  spirv.SpecConstant @sc3 = 1.5 : f32

  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.struct<(i1, i64, f32)>

  // CHECK-LABEL: @reference
  spirv.func @reference() -> i1 "None" {
    // CHECK: spirv.mlir.referenceof @sc1 : i1
    %0 = spirv.mlir.referenceof @sc1 : i1
    spirv.ReturnValue %0 : i1
  }

  // CHECK-LABEL: @reference_composite
  spirv.func @reference_composite() -> i1 "None" {
    // CHECK: spirv.mlir.referenceof @scc : !spirv.struct<(i1, i64, f32)>
    %0 = spirv.mlir.referenceof @scc : !spirv.struct<(i1, i64, f32)>
    %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.struct<(i1, i64, f32)>
    spirv.ReturnValue %1 : i1
  }

  // CHECK-LABEL: @initialize
  spirv.func @initialize() -> i64 "None" {
    // CHECK: spirv.mlir.referenceof @sc2 : i64
    %0 = spirv.mlir.referenceof @sc2 : i64
    %1 = spirv.Variable init(%0) : !spirv.ptr<i64, Function>
    %2 = spirv.Load "Function" %1 : i64
    spirv.ReturnValue %2 : i64
  }

  // CHECK-LABEL: @compute
  spirv.func @compute() -> f32 "None" {
    // CHECK: spirv.mlir.referenceof @sc3 : f32
    %0 = spirv.mlir.referenceof @sc3 : f32
    %1 = spirv.Constant 6.0 : f32
    %2 = spirv.FAdd %0, %1 : f32
    spirv.ReturnValue %2 : f32
  }
}

// -----

// Allow taking reference of spec constant in other module-like ops
spirv.SpecConstant @sc = 5 : i32
func.func @reference_of() {
  // CHECK: spirv.mlir.referenceof @sc
  %0 = spirv.mlir.referenceof @sc : i32
  return
}

// -----

spirv.SpecConstant @sc = 5 : i32
spirv.SpecConstantComposite @scc (@sc) : !spirv.array<1 x i32>

func.func @reference_of_composite() {
  // CHECK: spirv.mlir.referenceof @scc : !spirv.array<1 x i32>
  %0 = spirv.mlir.referenceof @scc : !spirv.array<1 x i32>
  %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.array<1 x i32>
  return
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @foo() -> () "None" {
    // expected-error @+1 {{expected spirv.SpecConstant or spirv.SpecConstantComposite symbol}}
    %0 = spirv.mlir.referenceof @sc : i32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc = 42 : i32
  spirv.func @foo() -> () "None" {
    // expected-error @+1 {{result type mismatch with the referenced specialization constant's type}}
    %0 = spirv.mlir.referenceof @sc : f32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc = 42 : i32
  spirv.SpecConstantComposite @scc (@sc) : !spirv.array<1 x i32>
  spirv.func @foo() -> () "None" {
    // expected-error @+1 {{result type mismatch with the referenced specialization constant's type}}
    %0 = spirv.mlir.referenceof @scc : f32
    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SpecConstant
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  // CHECK: spirv.SpecConstant @sc1 = false
  spirv.SpecConstant @sc1 = false
  // CHECK: spirv.SpecConstant @sc2 spec_id(5) = 42 : i64
  spirv.SpecConstant @sc2 spec_id(5) = 42 : i64
  // CHECK: spirv.SpecConstant @sc3 = 1.500000e+00 : f32
  spirv.SpecConstant @sc3 = 1.5 : f32
}

// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{SpecId cannot be negative}}
  spirv.SpecConstant @sc2 spec_id(-5) = 42 : i64
}

// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{default value bitwidth disallowed}}
  spirv.SpecConstant @sc = 15 : i4
}

// -----

spirv.module Logical GLSL450 {
  // expected-error @+1 {{default value can only be a bool, integer, or float scalar}}
  spirv.SpecConstant @sc = dense<[2, 3]> : vector<2xi32>
}

// -----

func.func @use_in_function() -> () {
  // expected-error @+1 {{op must appear in a module-like op's block}}
  spirv.SpecConstant @sc = false
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SpecConstantComposite
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  // expected-error @+1 {{result type must be a composite type}}
  spirv.SpecConstantComposite @scc2 (@sc1, @sc2, @sc3) : i32
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstantComposite (spirv.array)
//===----------------------------------------------------------------------===//

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1.5 : f32
  spirv.SpecConstant @sc2 = 2.5 : f32
  spirv.SpecConstant @sc3 = 3.5 : f32
  // CHECK: spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.array<3 x f32>
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.array<3 x f32>
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = false
  spirv.SpecConstant @sc2 spec_id(5) = 42 : i64
  spirv.SpecConstant @sc3 = 1.5 : f32
  // expected-error @+1 {{has incorrect number of operands: expected 4, but provided 3}}
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.array<4 x f32>

}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1   : i32
  spirv.SpecConstant @sc2 = 2.5 : f32
  spirv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect types of operands: expected 'f32', but provided 'i32'}}
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.array<3 x f32>
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstantComposite (spirv.struct)
//===----------------------------------------------------------------------===//

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1   : i32
  spirv.SpecConstant @sc2 = 2.5 : f32
  spirv.SpecConstant @sc3 = 3.5 : f32
  // CHECK: spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.struct<(i32, f32, f32)>
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.struct<(i32, f32, f32)>
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1   : i32
  spirv.SpecConstant @sc2 = 2.5 : f32
  spirv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect number of operands: expected 2, but provided 3}}
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.struct<(i32, f32)>
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1.5 : f32
  spirv.SpecConstant @sc2 = 2.5 : f32
  spirv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect types of operands: expected 'i32', but provided 'f32'}}
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.struct<(i32, f32, f32)>
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstantComposite (vector)
//===----------------------------------------------------------------------===//

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1.5 : f32
  spirv.SpecConstant @sc2 = 2.5 : f32
  spirv.SpecConstant @sc3 = 3.5 : f32
  // CHECK: spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<3xf32>
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<3 x f32>
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = false
  spirv.SpecConstant @sc2 spec_id(5) = 42 : i64
  spirv.SpecConstant @sc3 = 1.5 : f32
  // expected-error @+1 {{has incorrect number of operands: expected 4, but provided 3}}
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<4xf32>

}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1   : i32
  spirv.SpecConstant @sc2 = 2.5 : f32
  spirv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect types of operands: expected 'f32', but provided 'i32'}}
  spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<3xf32>
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstantComposite (spirv.KHR.coopmatrix)
//===----------------------------------------------------------------------===//

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc1 = 1.5 : f32
  // expected-error @+1 {{unsupported composite type}}
  spirv.SpecConstantComposite @scc (@sc1) : !spirv.coopmatrix<8x16xf32, Device, MatrixA>
}

//===----------------------------------------------------------------------===//
// spirv.SpecConstantOperation
//===----------------------------------------------------------------------===//

// -----

spirv.module Logical GLSL450 {
  spirv.func @foo() -> i32 "None" {
    // CHECK: [[LHS:%.*]] = spirv.Constant
    %0 = spirv.Constant 1: i32
    // CHECK: [[RHS:%.*]] = spirv.Constant
    %1 = spirv.Constant 1: i32

    // CHECK: spirv.SpecConstantOperation wraps "spirv.IAdd"([[LHS]], [[RHS]]) : (i32, i32) -> i32
    %2 = spirv.SpecConstantOperation wraps "spirv.IAdd"(%0, %1) : (i32, i32) -> i32

    spirv.ReturnValue %2 : i32
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.SpecConstant @sc = 42 : i32

  spirv.func @foo() -> i32 "None" {
    // CHECK: [[SC:%.*]] = spirv.mlir.referenceof @sc
    %0 = spirv.mlir.referenceof @sc : i32
    // CHECK: spirv.SpecConstantOperation wraps "spirv.ISub"([[SC]], [[SC]]) : (i32, i32) -> i32
    %1 = spirv.SpecConstantOperation wraps "spirv.ISub"(%0, %0) : (i32, i32) -> i32
    spirv.ReturnValue %1 : i32
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @foo() -> i32 "None" {
    %0 = spirv.Constant 1: i32
    // expected-error @+1 {{op expects parent op 'spirv.SpecConstantOperation'}}
    spirv.mlir.yield %0 : i32
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @foo() -> () "None" {
    %0 = spirv.Variable : !spirv.ptr<i32, Function>

    // expected-error @+1 {{invalid enclosed op}}
    %1 = spirv.SpecConstantOperation wraps "spirv.Load"(%0) {memory_access = #spirv.memory_access<None>} : (!spirv.ptr<i32, Function>) -> i32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @foo() -> () "None" {
    %0 = spirv.Variable : !spirv.ptr<i32, Function>
    %1 = spirv.Load "Function" %0 : i32

    // expected-error @+1 {{invalid operand, must be defined by a constant operation}}
    %2 = spirv.SpecConstantOperation wraps "spirv.IAdd"(%1, %1) : (i32, i32) -> i32

    spirv.Return
  }
}
