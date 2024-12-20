// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="bufferize-function-boundaries=1" -split-input-file -verify-diagnostics

func.func @scf_for(%A : tensor<?xf32>,
              %B : tensor<?xf32> {bufferization.writable = true},
              %C : tensor<4xf32>,
              %lb : index, %ub : index, %step : index)
  -> (f32, f32)
{
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // Throw a wrench in the system by swapping yielded values: this result in a
    // ping-pong of values at each iteration on which we currently want to fail.

    // expected-error @+1 {{Yield operand #0 is not equivalent to the corresponding iter bbArg}}
    scf.yield %ttB, %ttA : tensor<?xf32>, tensor<?xf32>
  }

  %f0 = tensor.extract %r0#0[%step] : tensor<?xf32>
  %f1 = tensor.extract %r0#1[%step] : tensor<?xf32>
  return %f0, %f1: f32, f32
}

// -----

func.func @scf_while_non_equiv_condition(%arg0: tensor<5xi1>,
                                         %arg1: tensor<5xi1>,
                                         %idx: index) -> (i1, i1)
{
  %r0, %r1 = scf.while (%w0 = %arg0, %w1 = %arg1)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    // expected-error @+1 {{Condition arg #0 is not equivalent to the corresponding iter bbArg}}
    scf.condition(%condition) %w1, %w0 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %b0[%pos] : tensor<5xi1>
    scf.yield %1, %b1 : tensor<5xi1>, tensor<5xi1>
  }

  %v0 = tensor.extract %r0[%idx] : tensor<5xi1>
  %v1 = tensor.extract %r1[%idx] : tensor<5xi1>
  return %v0, %v1 : i1, i1
}

// -----

func.func @scf_while_non_equiv_yield(%arg0: tensor<5xi1>,
                                     %arg1: tensor<5xi1>,
                                     %idx: index) -> (i1, i1)
{
  %r0, %r1 = scf.while (%w0 = %arg0, %w1 = %arg1)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    scf.condition(%condition) %w0, %w1 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %b0[%pos] : tensor<5xi1>
    // expected-error @+1 {{Yield operand #0 is not equivalent to the corresponding iter bbArg}}
    scf.yield %b1, %1 : tensor<5xi1>, tensor<5xi1>
  }

  %v0 = tensor.extract %r0[%idx] : tensor<5xi1>
  %v1 = tensor.extract %r1[%idx] : tensor<5xi1>
  return %v0, %v1 : i1, i1
}

// -----

func.func @to_tensor_op_unsupported(%m: memref<?xf32>, %idx: index) -> (f32) {
  // expected-error @+1 {{to_tensor ops without `restrict` are not supported by One-Shot Analysis}}
  %0 = bufferization.to_tensor %m : memref<?xf32> to tensor<?xf32>

  %1 = tensor.extract %0[%idx] : tensor<?xf32>
  return %1 : f32
}

// -----

func.func @yield_alloc_dominance_test_2(%cst : f32, %idx : index,
                                        %idx2 : index) -> f32 {
  %1 = bufferization.alloc_tensor(%idx) : tensor<?xf32>

  %0 = scf.execute_region -> tensor<?xf32> {
    // This YieldOp returns a value that is defined in a parent block, thus
    // no error.
    scf.yield %1 : tensor<?xf32>
  }
  %2 = tensor.insert %cst into %0[%idx] : tensor<?xf32>
  %r = tensor.extract %2[%idx2] : tensor<?xf32>
  return %r : f32
}

// -----

func.func @copy_of_unranked_tensor(%t: tensor<*xf32>) -> tensor<*xf32> {
  // Unranked tensor OpOperands always bufferize in-place. With this limitation,
  // there is no way to bufferize this IR correctly.
  // expected-error @+1 {{not bufferizable under the given constraints: cannot avoid RaW conflict}}
  func.call @maybe_writing_func(%t) : (tensor<*xf32>) -> ()
  return %t : tensor<*xf32>
}

// This function may write to buffer(%ptr).
func.func private @maybe_writing_func(%ptr : tensor<*xf32>)

// -----

func.func @regression_scf_while() {
  %false = arith.constant false
  %8 = bufferization.alloc_tensor() : tensor<10x10xf32>
  scf.while (%arg0 = %8) : (tensor<10x10xf32>) -> () {
    scf.condition(%false)
  } do {
    // expected-error @+1 {{Yield operand #0 is not equivalent to the corresponding iter bbArg}}
    scf.yield %8 : tensor<10x10xf32>
  }
  return
}

// -----

// expected-error @below{{could not infer buffer type of block argument}}
// expected-error @below{{failed to bufferize op}}
func.func @func_multiple_yields(%t: tensor<5xf32>) -> tensor<5xf32> {
  func.return %t : tensor<5xf32>
^bb1(%arg1 : tensor<5xf32>):
  func.return %arg1 : tensor<5xf32>
}
