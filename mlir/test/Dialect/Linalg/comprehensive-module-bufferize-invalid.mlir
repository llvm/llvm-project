// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize -split-input-file -verify-diagnostics

func private @foo() -> tensor<?xf32>

func @bar() -> tensor<?xf32> {
  %foo = constant @foo : () -> (tensor<?xf32>)
// expected-error @+1 {{expected a CallOp}}
  %res = call_indirect %foo() : () -> (tensor<?xf32>)
  return %res : tensor<?xf32>
}

// -----

// expected-error @+1 {{cannot bufferize bodiless function that returns a tensor}}
func private @foo() -> tensor<?xf32>

// -----

// expected-error @+1 {{cannot bufferize a FuncOp with tensors and without a unique ReturnOp}}
func @switch(%flag : i32, %caseOperand : i32, %t1 : tensor<f32>, %t2 : tensor<f32>)
    -> (tensor<f32>) 
{
  switch %flag : i32, [
    default: ^bb1(%caseOperand : i32),
    42: ^bb2(%caseOperand : i32)
  ]

  ^bb1(%bb1arg : i32):
    return %t1 : tensor<f32>
  ^bb2(%bb2arg : i32):
    return %t2 : tensor<f32>
}

// -----

// expected-error @-3 {{expected callgraph to be free of circular dependencies}}

func @foo() {
  call @bar() : () -> ()
  return
}

func @bar() {
  call @foo() : () -> ()
  return
}
