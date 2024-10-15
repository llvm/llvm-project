// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-scf-to-cf), func.func(test-print-liveness))"

module {
  func.func @for_if_for(%arg0: index, %arg1: index, %arg2: index, %arg3: i1) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
    %0 = scf.for %arg4 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %cst) -> (tensor<128x32xf16>) {
      %1 = scf.if %arg3 -> (tensor<128x32xf16>) {
        scf.yield %arg5 : tensor<128x32xf16>
      } else {
        %2 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%arg7 = %arg5) -> (tensor<128x32xf16>) {
          scf.yield %arg7 : tensor<128x32xf16>
        }
        scf.yield %2 : tensor<128x32xf16>
      }
      scf.yield %1 : tensor<128x32xf16>
    }
    
    return
  }
}