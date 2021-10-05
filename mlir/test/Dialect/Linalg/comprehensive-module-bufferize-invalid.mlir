// RUN: mlir-opt %s -allow-unregistered-dialect -linalg-comprehensive-module-bufferize -split-input-file -verify-diagnostics

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
func @swappy(%cond1 : i1, %cond2 : i1, %t1 : tensor<f32>, %t2 : tensor<f32>)
    -> (tensor<f32>, tensor<f32>)
{
  cond_br %cond1, ^bb1, ^bb2

  ^bb1:
    %T:2 = scf.if %cond2 -> (tensor<f32>, tensor<f32>) {
      scf.yield %t1, %t2 : tensor<f32>, tensor<f32>
    } else {
      scf.yield %t2, %t1 : tensor<f32>, tensor<f32>
    }
    return %T#0, %T#1 : tensor<f32>, tensor<f32>
  ^bb2:
    return %t2, %t1 : tensor<f32>, tensor<f32>
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

// -----

func @scf_for(%A : tensor<?xf32>,
              %B : tensor<?xf32> {linalg.inplaceable = true},
              %C : tensor<4xf32>,
              %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // Throw a wrench in the system by swapping yielded values: this result in a
    // ping-pong of values at each iteration on which we currently want to fail.

    // expected-error @+1 {{Yield operand #0 does not bufferize to an equivalent buffer}}
    scf.yield %ttB, %ttA : tensor<?xf32>, tensor<?xf32>
  }

  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

func private @fun_with_side_effects(%A: tensor<?xf32> {linalg.inplaceable = true})

func @foo(%A: tensor<?xf32> {linalg.inplaceable = true}) -> (tensor<?xf32>) {
  call @fun_with_side_effects(%A) : (tensor<?xf32>) -> ()
  return %A: tensor<?xf32>
}

func @scf_yield_needs_copy(%A : tensor<?xf32> {linalg.inplaceable = true}, %iters : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %res = scf.for %arg0 = %c0 to %iters step %c1 iter_args(%bbarg = %A) -> (tensor<?xf32>) {
    %r = call @foo(%A) : (tensor<?xf32>) -> (tensor<?xf32>)
    // expected-error @+1 {{Yield operand #0 does not bufferize to an equivalent buffer}}
    scf.yield %r : tensor<?xf32>
  }
  call @fun_with_side_effects(%res) : (tensor<?xf32>) -> ()
  return
}

// -----

func @extract_slice_fun(%A : tensor<?xf32> {linalg.inplaceable = true})
  ->  tensor<4xf32>
{
  // This bufferizes to a pattern that the cross-function boundary pass needs to
  // convert into a new memref argument at all call site; this may be either:
  //   - an externally created aliasing subview (if we want to allow aliasing
  //     function arguments).
  //   - a new alloc + copy (more expensive but does not create new function
  //     argument aliasing).
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // expected-error @+1 {{buffer result #0 not produced by an alloc}}
  return %r0: tensor<4xf32>
}

// -----

func @scf_yield(%b : i1, %A : tensor<4xf32>, %B : tensor<4xf32>) -> tensor<4xf32>
{
  // expected-error @+1 {{unsupported op with tensors}}
  %r = scf.if %b -> (tensor<4xf32>) {
    scf.yield %A : tensor<4xf32>
  } else {
    scf.yield %B : tensor<4xf32>
  }
  return %r: tensor<4xf32>
}

// -----

func @unknown_op(%A : tensor<4xf32>) -> tensor<4xf32>
{
  // expected-error @+1 {{unsupported op with tensors}}
  %r = "marklar"(%A) : (tensor<4xf32>) -> (tensor<4xf32>)
  return %r: tensor<4xf32>
}

// -----

// expected-error @+1 {{memref return type is unsupported}}
func @mini_test_case1() -> tensor<10x20xf32> {
  %f0 = constant 0.0 : f32
  %t = linalg.init_tensor [10, 20] : tensor<10x20xf32>
  %r = linalg.fill(%f0, %t) : f32, tensor<10x20xf32> -> tensor<10x20xf32>
  return %r : tensor<10x20xf32>
}

// -----

func @main() -> tensor<4xi32> {
  // expected-error @+1 {{unsupported op with tensors}}
  %r = scf.execute_region -> tensor<4xi32> {
    %A = constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    scf.yield %A: tensor<4xi32>
  }
  return %r: tensor<4xi32>
}

// -----

func @main() -> i32 {
  %c0 = constant 0: index
  // expected-error @+1 {{expected result-less scf.execute_region containing op}}
  %r = scf.execute_region -> i32 {
    %A = constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    %e = tensor.extract %A[%c0]: tensor<4xi32>
    scf.yield %e: i32
  }
  return %r: i32
}
