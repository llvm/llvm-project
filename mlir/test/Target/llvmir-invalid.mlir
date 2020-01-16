// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// expected-error @+1 {{unsupported module-level operation}}
func @foo() {
  llvm.return
}

// -----

llvm.func @no_nested_struct() -> !llvm<"[2 x [2 x [2 x {i32}]]]"> {
  // expected-error @+1 {{struct types are not supported in constants}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm<"[2 x [2 x [2 x {i32}]]]">
  llvm.return %0 : !llvm<"[2 x [2 x [2 x {i32}]]]">
}
