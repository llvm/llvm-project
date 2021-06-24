// RUN: mlir-opt %s -linalg-comprehensive-func-bufferize=test-analysis-only -split-input-file -verify-diagnostics

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

    // expected-error @+1 {{Yield operand #1 does not bufferize to an equivalent buffer}}
    scf.yield %ttB, %ttA : tensor<?xf32>, tensor<?xf32>
  }

  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

