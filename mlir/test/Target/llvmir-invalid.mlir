// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// expected-error @+1 {{unsupported module-level operation}}
func @foo() {
  llvm.return
}

// -----

// expected-error @+1 {{llvm.noalias attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_noalias(%arg0 : !llvm.float {llvm.noalias = true}) -> !llvm.float {
  llvm.return %arg0 : !llvm.float
}

// -----

// expected-error @+1 {{llvm.align attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_align(%arg0 : !llvm.float {llvm.align = 4}) -> !llvm.float {
  llvm.return %arg0 : !llvm.float
}

// -----

llvm.func @no_nested_struct() -> !llvm<"[2 x [2 x [2 x {i32}]]]"> {
  // expected-error @+1 {{struct types are not supported in constants}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm<"[2 x [2 x [2 x {i32}]]]">
  llvm.return %0 : !llvm<"[2 x [2 x [2 x {i32}]]]">
}

// -----

// expected-error @+1 {{unsupported constant value}}
llvm.mlir.global internal constant @test([2.5, 7.4]) : !llvm<"[2 x double]">

// -----

// expected-error @+1 {{LLVM attribute 'noinline' does not expect a value}}
llvm.func @passthrough_unexpected_value() attributes {passthrough = [["noinline", "42"]]}

// -----

// expected-error @+1 {{LLVM attribute 'alignstack' expects a value}}
llvm.func @passthrough_expected_value() attributes {passthrough = ["alignstack"]}

// -----

// expected-error @+1 {{expected 'passthrough' to contain string or array attributes}}
llvm.func @passthrough_wrong_type() attributes {passthrough = [42]}

// -----

// expected-error @+1 {{expected arrays within 'passthrough' to contain two strings}}
llvm.func @passthrough_wrong_type() attributes {passthrough = [[42, 42]]}
