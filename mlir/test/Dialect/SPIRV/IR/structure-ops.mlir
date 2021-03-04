// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.mlir.addressof
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>
  spv.func @access_chain() -> () "None" {
    %0 = spv.constant 1: i32
    // CHECK: [[VAR1:%.*]] = spv.mlir.addressof @var1 : !spv.ptr<!spv.struct<(f32, !spv.array<4 x f32>)>, Input>
    // CHECK-NEXT: spv.AccessChain [[VAR1]][{{.*}}, {{.*}}] : !spv.ptr<!spv.struct<(f32, !spv.array<4 x f32>)>, Input>
    %1 = spv.mlir.addressof @var1 : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>
    %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>, i32, i32
    spv.Return
  }
}

// -----

// Allow taking address of global variables in other module-like ops
spv.globalVariable @var : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>
func @addressof() -> () {
  // CHECK: spv.mlir.addressof @var
  %1 = spv.mlir.addressof @var : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>
  return
}

// -----

spv.module Logical GLSL450 {
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>
  spv.func @foo() -> () "None" {
    // expected-error @+1 {{expected spv.globalVariable symbol}}
    %0 = spv.mlir.addressof @var2 : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>
  }
}

// -----

spv.module Logical GLSL450 {
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<(f32, !spv.array<4xf32>)>, Input>
  spv.func @foo() -> () "None" {
    // expected-error @+1 {{result type mismatch with the referenced global variable's type}}
    %0 = spv.mlir.addressof @var1 : !spv.ptr<f32, Input>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

func @const() -> () {
  // CHECK: spv.constant true
  // CHECK: spv.constant 42 : i32
  // CHECK: spv.constant 5.000000e-01 : f32
  // CHECK: spv.constant dense<[2, 3]> : vector<2xi32>
  // CHECK: spv.constant [dense<3.000000e+00> : vector<2xf32>] : !spv.array<1 x vector<2xf32>>
  // CHECK: spv.constant dense<1> : tensor<2x3xi32> : !spv.array<2 x !spv.array<3 x i32>>
  // CHECK: spv.constant dense<1.000000e+00> : tensor<2x3xf32> : !spv.array<2 x !spv.array<3 x f32>>
  // CHECK: spv.constant dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32> : !spv.array<2 x !spv.array<3 x i32>>
  // CHECK: spv.constant dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32> : !spv.array<2 x !spv.array<3 x f32>>

  %0 = spv.constant true
  %1 = spv.constant 42 : i32
  %2 = spv.constant 0.5 : f32
  %3 = spv.constant dense<[2, 3]> : vector<2xi32>
  %4 = spv.constant [dense<3.0> : vector<2xf32>] : !spv.array<1xvector<2xf32>>
  %5 = spv.constant dense<1> : tensor<2x3xi32> : !spv.array<2 x !spv.array<3 x i32>>
  %6 = spv.constant dense<1.0> : tensor<2x3xf32> : !spv.array<2 x !spv.array<3 x f32>>
  %7 = spv.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32> : !spv.array<2 x !spv.array<3 x i32>>
  %8 = spv.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32> : !spv.array<2 x !spv.array<3 x f32>>
  return
}

// -----

func @unaccepted_std_attr() -> () {
  // expected-error @+1 {{cannot have value of type 'none'}}
  %0 = spv.constant unit : none
  return
}

// -----

func @array_constant() -> () {
  // expected-error @+1 {{has array element whose type ('vector<2xi32>') does not match the result element type ('vector<2xf32>')}}
  %0 = spv.constant [dense<3.0> : vector<2xf32>, dense<4> : vector<2xi32>] : !spv.array<2xvector<2xf32>>
  return
}

// -----

func @array_constant() -> () {
  // expected-error @+1 {{must have spv.array result type for array value}}
  %0 = spv.constant [dense<3.0> : vector<2xf32>] : !spv.rtarray<vector<2xf32>>
  return
}

// -----

func @non_nested_array_constant() -> () {
  // expected-error @+1 {{only support nested array result type}}
  %0 = spv.constant dense<3.0> : tensor<2x2xf32> : !spv.array<2xvector<2xf32>>
  return
}

// -----

func @value_result_type_mismatch() -> () {
  // expected-error @+1 {{must have spv.array result type for array value}}
  %0 = "spv.constant"() {value = dense<0> : tensor<4xi32>} : () -> (vector<4xi32>)
}

// -----

func @value_result_type_mismatch() -> () {
  // expected-error @+1 {{result element type ('i32') does not match value element type ('f32')}}
  %0 = spv.constant dense<1.0> : tensor<2x3xf32> : !spv.array<2 x !spv.array<3 x i32>>
}

// -----

func @value_result_num_elements_mismatch() -> () {
  // expected-error @+1 {{result number of elements (6) does not match value number of elements (4)}}
  %0 = spv.constant dense<1.0> : tensor<2x2xf32> : !spv.array<2 x !spv.array<3 x f32>>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.EntryPoint
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   // CHECK: spv.EntryPoint "GLCompute" @do_nothing
   spv.EntryPoint "GLCompute" @do_nothing
}

spv.module Logical GLSL450 {
   spv.globalVariable @var2 : !spv.ptr<f32, Input>
   spv.globalVariable @var3 : !spv.ptr<f32, Output>
   spv.func @do_something(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) -> () "None" {
     %1 = spv.Load "Input" %arg0 : f32
     spv.Store "Output" %arg1, %1 : f32
     spv.Return
   }
   // CHECK: spv.EntryPoint "GLCompute" @do_something, @var2, @var3
   spv.EntryPoint "GLCompute" @do_something, @var2, @var3
}

// -----

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   // expected-error @+1 {{invalid kind of attribute specified}}
   spv.EntryPoint "GLCompute" "do_nothing"
}

// -----

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   // expected-error @+1 {{function 'do_something' not found in 'spv.module'}}
   spv.EntryPoint "GLCompute" @do_something
}

/// TODO: Add a test that verifies an error is thrown
/// when interface entries of EntryPointOp are not
/// spv.Variables. There is currently no other op that has a spv.ptr
/// return type

// -----

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     // expected-error @+1 {{op must appear in a module-like op's block}}
     spv.EntryPoint "GLCompute" @do_something
   }
}

// -----

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{duplicate of a previous EntryPointOp}}
   spv.EntryPoint "GLCompute" @do_nothing
}

// -----

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{'spv.EntryPoint' invalid execution_model attribute specification: "ContractionOff"}}
   spv.EntryPoint "ContractionOff" @do_nothing
}

// -----

//===----------------------------------------------------------------------===//
// spv.ExecutionMode
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spv.ExecutionMode {{@.*}} "ContractionOff"
   spv.ExecutionMode @do_nothing "ContractionOff"
}

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spv.ExecutionMode {{@.*}} "LocalSizeHint", 3, 4, 5
   spv.ExecutionMode @do_nothing "LocalSizeHint", 3, 4, 5
}

// -----

spv.module Logical GLSL450 {
   spv.func @do_nothing() -> () "None" {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{custom op 'spv.ExecutionMode' invalid execution_mode attribute specification: "GLCompute"}}
   spv.ExecutionMode @do_nothing "GLCompute", 3, 4, 5
}

// -----

//===----------------------------------------------------------------------===//
// spv.func
//===----------------------------------------------------------------------===//

// CHECK: spv.func @foo() "None"
spv.func @foo() "None"

// CHECK: spv.func @bar(%{{.+}}: i32) -> i32 "Inline|Pure" {
spv.func @bar(%arg: i32) -> (i32) "Inline|Pure" {
  // CHECK-NEXT: spv.
  spv.ReturnValue %arg: i32
// CHECK-NEXT: }
}

// CHECK: spv.func @baz(%{{.+}}: i32) "DontInline" attributes {additional_stuff = 64 : i64}
spv.func @baz(%arg: i32) "DontInline" attributes {
  additional_stuff = 64
} { spv.Return }

// -----

// expected-error @+1 {{expected function_control attribute specified as string}}
spv.func @missing_function_control() { spv.Return }

// -----

// expected-error @+1 {{cannot have more than one result}}
spv.func @cannot_have_more_than_one_result(%arg: i32) -> (i32, i32) "None"

// -----

// expected-error @+1 {{expected SSA identifier}}
spv.func @cannot_have_variadic_arguments(%arg: i32, ...) "None"

// -----

// Nested function
spv.module Logical GLSL450 {
  spv.func @outer_func() -> () "None" {
    // expected-error @+1 {{must appear in a module-like op's block}}
    spv.func @inner_func() -> () "None" {
      spv.Return
    }
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.globalVariable
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  // CHECK: spv.globalVariable @var0 : !spv.ptr<f32, Input>
  spv.globalVariable @var0 : !spv.ptr<f32, Input>
}

// TODO: Fix test case after initialization with normal constant is addressed
// spv.module Logical GLSL450 {
//   %0 = spv.constant 4.0 : f32
//   // CHECK1: spv.Variable init(%0) : !spv.ptr<f32, Private>
//   spv.globalVariable @var1 init(%0) : !spv.ptr<f32, Private>
// }

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc = 4.0 : f32
  // CHECK: spv.globalVariable @var initializer(@sc) : !spv.ptr<f32, Private>
  spv.globalVariable @var initializer(@sc) : !spv.ptr<f32, Private>
}

// -----

// Allow initializers coming from other module-like ops
spv.SpecConstant @sc = 4.0 : f32
// CHECK: spv.globalVariable @var initializer(@sc)
spv.globalVariable @var initializer(@sc) : !spv.ptr<f32, Private>

// -----

spv.module Logical GLSL450 {
  // CHECK: spv.globalVariable @var0 bind(1, 2) : !spv.ptr<f32, Uniform>
  spv.globalVariable @var0 bind(1, 2) : !spv.ptr<f32, Uniform>
}

// TODO: Fix test case after initialization with constant is addressed
// spv.module Logical GLSL450 {
//   %0 = spv.constant 4.0 : f32
//   // CHECK1: spv.globalVariable @var1 initializer(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
//   spv.globalVariable @var1 initializer(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
// }

// -----

spv.module Logical GLSL450 {
  // CHECK: spv.globalVariable @var1 built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @var1 built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable @var2 built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @var2 {built_in = "GlobalInvocationID"} : !spv.ptr<vector<3xi32>, Input>
}

// -----

// Allow in other module-like ops
module {
  // CHECK: spv.globalVariable
  spv.globalVariable @var0 : !spv.ptr<f32, Input>
}

// -----

spv.module Logical GLSL450 {
  // expected-error @+1 {{expected spv.ptr type}}
  spv.globalVariable @var0 : f32
}

// -----

spv.module Logical GLSL450 {
  // expected-error @+1 {{op initializer must be result of a spv.SpecConstant or spv.globalVariable op}}
  spv.globalVariable @var0 initializer(@var1) : !spv.ptr<f32, Private>
}

// -----

spv.module Logical GLSL450 {
  // expected-error @+1 {{storage class cannot be 'Generic'}}
  spv.globalVariable @var0 : !spv.ptr<f32, Generic>
}

// -----

spv.module Logical GLSL450 {
  // expected-error @+1 {{storage class cannot be 'Function'}}
  spv.globalVariable @var0 : !spv.ptr<f32, Function>
}

// -----

spv.module Logical GLSL450 {
  spv.func @foo() "None" {
    // expected-error @+1 {{op must appear in a module-like op's block}}
    spv.globalVariable @var0 : !spv.ptr<f32, Input>
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

// Module without capability and extension
// CHECK: spv.module Logical GLSL450
spv.module Logical GLSL450 { }

// Module with a name
// CHECK: spv.module @{{.*}} Logical GLSL450
spv.module @name Logical GLSL450 { }

// Module with (version, capabilities, extensions) triple
// CHECK: spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]>
spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]> { }

// Module with additional attributes
// CHECK: spv.module Logical GLSL450 attributes {foo = "bar"}
spv.module Logical GLSL450 attributes {foo = "bar"} { }

// Module with VCE triple and additional attributes
// CHECK: spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]> attributes {foo = "bar"}
spv.module Logical GLSL450
  requires #spv.vce<v1.0, [Shader], [SPV_KHR_16bit_storage]>
  attributes {foo = "bar"} { }

// Module with explicit spv.mlir.endmodule
// CHECK: spv.module
spv.module Logical GLSL450 {
  spv.mlir.endmodule
}

// Module with function
// CHECK: spv.module
spv.module Logical GLSL450 {
  spv.func @do_nothing() -> () "None" {
    spv.Return
  }
}

// -----

// Missing addressing model
// expected-error@+1 {{'spv.module' expected valid keyword}}
spv.module { }

// -----

// Wrong addressing model
// expected-error@+1 {{'spv.module' invalid addressing_model attribute specification: Physical}}
spv.module Physical { }

// -----

// Missing memory model
// expected-error@+1 {{'spv.module' expected valid keyword}}
spv.module Logical { }

// -----

// Wrong memory model
// expected-error@+1 {{'spv.module' invalid memory_model attribute specification: Bla}}
spv.module Logical Bla { }

// -----

// Module with multiple blocks
// expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
spv.module Logical GLSL450 {
^first:
  spv.Return
^second:
  spv.Return
}

// -----

// Module with wrong terminator
// expected-error@+2 {{expects regions to end with 'spv.mlir.endmodule'}}
// expected-note@+1 {{in custom textual format, the absence of terminator implies 'spv.mlir.endmodule'}}
"spv.module"() ({
  %0 = spv.constant true
}) {addressing_model = 0 : i32, memory_model = 1 : i32} : () -> ()

// -----

// Use non SPIR-V op inside module
spv.module Logical GLSL450 {
  // expected-error @+1 {{'spv.module' can only contain spv.* ops}}
  "dialect.op"() : () -> ()
}

// -----

// Use non SPIR-V op inside function
spv.module Logical GLSL450 {
  spv.func @do_nothing() -> () "None" {
    // expected-error @+1 {{functions in 'spv.module' can only contain spv.* ops}}
    "dialect.op"() : () -> ()
  }
}

// -----

// Use external function
spv.module Logical GLSL450 {
  // expected-error @+1 {{'spv.module' cannot contain external functions}}
  spv.func @extern() -> () "None"
}

// -----

//===----------------------------------------------------------------------===//
// spv.mlir.endmodule
//===----------------------------------------------------------------------===//

func @module_end_not_in_module() -> () {
  // expected-error @+1 {{op must appear in a module-like op's block}}
  spv.mlir.endmodule
}

// -----

//===----------------------------------------------------------------------===//
// spv.mlir.referenceof
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = false
  spv.SpecConstant @sc2 = 42 : i64
  spv.SpecConstant @sc3 = 1.5 : f32

  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.struct<(i1, i64, f32)>

  // CHECK-LABEL: @reference
  spv.func @reference() -> i1 "None" {
    // CHECK: spv.mlir.referenceof @sc1 : i1
    %0 = spv.mlir.referenceof @sc1 : i1
    spv.ReturnValue %0 : i1
  }

  // CHECK-LABEL: @reference_composite
  spv.func @reference_composite() -> i1 "None" {
    // CHECK: spv.mlir.referenceof @scc : !spv.struct<(i1, i64, f32)>
    %0 = spv.mlir.referenceof @scc : !spv.struct<(i1, i64, f32)>
    %1 = spv.CompositeExtract %0[0 : i32] : !spv.struct<(i1, i64, f32)>
    spv.ReturnValue %1 : i1
  }

  // CHECK-LABEL: @initialize
  spv.func @initialize() -> i64 "None" {
    // CHECK: spv.mlir.referenceof @sc2 : i64
    %0 = spv.mlir.referenceof @sc2 : i64
    %1 = spv.Variable init(%0) : !spv.ptr<i64, Function>
    %2 = spv.Load "Function" %1 : i64
    spv.ReturnValue %2 : i64
  }

  // CHECK-LABEL: @compute
  spv.func @compute() -> f32 "None" {
    // CHECK: spv.mlir.referenceof @sc3 : f32
    %0 = spv.mlir.referenceof @sc3 : f32
    %1 = spv.constant 6.0 : f32
    %2 = spv.FAdd %0, %1 : f32
    spv.ReturnValue %2 : f32
  }
}

// -----

// Allow taking reference of spec constant in other module-like ops
spv.SpecConstant @sc = 5 : i32
func @reference_of() {
  // CHECK: spv.mlir.referenceof @sc
  %0 = spv.mlir.referenceof @sc : i32
  return
}

// -----

spv.SpecConstant @sc = 5 : i32
spv.SpecConstantComposite @scc (@sc) : !spv.array<1 x i32>

func @reference_of_composite() {
  // CHECK: spv.mlir.referenceof @scc : !spv.array<1 x i32>
  %0 = spv.mlir.referenceof @scc : !spv.array<1 x i32>
  %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<1 x i32>
  return
}

// -----

spv.module Logical GLSL450 {
  spv.func @foo() -> () "None" {
    // expected-error @+1 {{expected spv.SpecConstant or spv.SpecConstantComposite symbol}}
    %0 = spv.mlir.referenceof @sc : i32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc = 42 : i32
  spv.func @foo() -> () "None" {
    // expected-error @+1 {{result type mismatch with the referenced specialization constant's type}}
    %0 = spv.mlir.referenceof @sc : f32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc = 42 : i32
  spv.SpecConstantComposite @scc (@sc) : !spv.array<1 x i32>
  spv.func @foo() -> () "None" {
    // expected-error @+1 {{result type mismatch with the referenced specialization constant's type}}
    %0 = spv.mlir.referenceof @scc : f32
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.SpecConstant
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  // CHECK: spv.SpecConstant @sc1 = false
  spv.SpecConstant @sc1 = false
  // CHECK: spv.SpecConstant @sc2 spec_id(5) = 42 : i64
  spv.SpecConstant @sc2 spec_id(5) = 42 : i64
  // CHECK: spv.SpecConstant @sc3 = 1.500000e+00 : f32
  spv.SpecConstant @sc3 = 1.5 : f32
}

// -----

spv.module Logical GLSL450 {
  // expected-error @+1 {{SpecId cannot be negative}}
  spv.SpecConstant @sc2 spec_id(-5) = 42 : i64
}

// -----

spv.module Logical GLSL450 {
  // expected-error @+1 {{default value bitwidth disallowed}}
  spv.SpecConstant @sc = 15 : i4
}

// -----

spv.module Logical GLSL450 {
  // expected-error @+1 {{default value can only be a bool, integer, or float scalar}}
  spv.SpecConstant @sc = dense<[2, 3]> : vector<2xi32>
}

// -----

func @use_in_function() -> () {
  // expected-error @+1 {{op must appear in a module-like op's block}}
  spv.SpecConstant @sc = false
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.SpecConstantComposite
//===----------------------------------------------------------------------===//

spv.module Logical GLSL450 {
  // expected-error @+1 {{result type must be a composite type}}
  spv.SpecConstantComposite @scc2 (@sc1, @sc2, @sc3) : i32
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantComposite (spv.array)
//===----------------------------------------------------------------------===//

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1.5 : f32
  spv.SpecConstant @sc2 = 2.5 : f32
  spv.SpecConstant @sc3 = 3.5 : f32
  // CHECK: spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.array<3 x f32>
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.array<3 x f32>
}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = false
  spv.SpecConstant @sc2 spec_id(5) = 42 : i64
  spv.SpecConstant @sc3 = 1.5 : f32
  // expected-error @+1 {{has incorrect number of operands: expected 4, but provided 3}}
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.array<4 x f32>

}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1   : i32
  spv.SpecConstant @sc2 = 2.5 : f32
  spv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect types of operands: expected 'f32', but provided 'i32'}}
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.array<3 x f32>
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantComposite (spv.struct)
//===----------------------------------------------------------------------===//

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1   : i32
  spv.SpecConstant @sc2 = 2.5 : f32
  spv.SpecConstant @sc3 = 3.5 : f32
  // CHECK: spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.struct<(i32, f32, f32)>
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.struct<(i32, f32, f32)>
}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1   : i32
  spv.SpecConstant @sc2 = 2.5 : f32
  spv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect number of operands: expected 2, but provided 3}}
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.struct<(i32, f32)>
}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1.5 : f32
  spv.SpecConstant @sc2 = 2.5 : f32
  spv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect types of operands: expected 'i32', but provided 'f32'}}
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.struct<(i32, f32, f32)>
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantComposite (vector)
//===----------------------------------------------------------------------===//

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1.5 : f32
  spv.SpecConstant @sc2 = 2.5 : f32
  spv.SpecConstant @sc3 = 3.5 : f32
  // CHECK: spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<3xf32>
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<3 x f32>
}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = false
  spv.SpecConstant @sc2 spec_id(5) = 42 : i64
  spv.SpecConstant @sc3 = 1.5 : f32
  // expected-error @+1 {{has incorrect number of operands: expected 4, but provided 3}}
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<4xf32>

}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1   : i32
  spv.SpecConstant @sc2 = 2.5 : f32
  spv.SpecConstant @sc3 = 3.5 : f32
  // expected-error @+1 {{has incorrect types of operands: expected 'f32', but provided 'i32'}}
  spv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : vector<3xf32>
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantComposite (spv.coopmatrix)
//===----------------------------------------------------------------------===//

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc1 = 1.5 : f32
  // expected-error @+1 {{unsupported composite type}}
  spv.SpecConstantComposite @scc (@sc1) : !spv.coopmatrix<8x16xf32, Device>
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantOperation
//===----------------------------------------------------------------------===//

// -----

spv.module Logical GLSL450 {
  spv.func @foo() -> i32 "None" {
    // CHECK: [[LHS:%.*]] = spv.constant
    %0 = spv.constant 1: i32
    // CHECK: [[RHS:%.*]] = spv.constant
    %1 = spv.constant 1: i32

    // CHECK: spv.SpecConstantOperation wraps "spv.IAdd"([[LHS]], [[RHS]]) : (i32, i32) -> i32
    %2 = spv.SpecConstantOperation wraps "spv.IAdd"(%0, %1) : (i32, i32) -> i32

    spv.ReturnValue %2 : i32
  }
}

// -----

spv.module Logical GLSL450 {
  spv.SpecConstant @sc = 42 : i32

  spv.func @foo() -> i32 "None" {
    // CHECK: [[SC:%.*]] = spv.mlir.referenceof @sc
    %0 = spv.mlir.referenceof @sc : i32
    // CHECK: spv.SpecConstantOperation wraps "spv.ISub"([[SC]], [[SC]]) : (i32, i32) -> i32
    %1 = spv.SpecConstantOperation wraps "spv.ISub"(%0, %0) : (i32, i32) -> i32
    spv.ReturnValue %1 : i32
  }
}

// -----

spv.module Logical GLSL450 {
  spv.func @foo() -> i32 "None" {
    %0 = spv.constant 1: i32
    // expected-error @+1 {{op expects parent op 'spv.SpecConstantOperation'}}
    spv.mlir.yield %0 : i32
  }
}

// -----

spv.module Logical GLSL450 {
  spv.func @foo() -> () "None" {
    %0 = spv.Variable : !spv.ptr<i32, Function>

    // expected-error @+1 {{invalid enclosed op}}
    %1 = spv.SpecConstantOperation wraps "spv.Load"(%0) {memory_access = 0 : i32} : (!spv.ptr<i32, Function>) -> i32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.func @foo() -> () "None" {
    %0 = spv.Variable : !spv.ptr<i32, Function>
    %1 = spv.Load "Function" %0 : i32

    // expected-error @+1 {{invalid operand, must be defined by a constant operation}}
    %2 = spv.SpecConstantOperation wraps "spv.IAdd"(%1, %1) : (i32, i32) -> i32

    spv.Return
  }
}
