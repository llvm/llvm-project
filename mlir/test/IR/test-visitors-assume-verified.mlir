// RUN: mlir-opt -test-ir-visitors --mlir-print-assume-verified %s | FileCheck %s

// Regression test: linalg ops implementing getAsmBlockArgumentNames via
// getRegionInputArgs() used to crash during block erasure in no-skip walks
// when combined with --mlir-print-assume-verified, because AsmState
// construction would call getAsmBlockArgumentNames on ops whose regions
// had already been emptied.

func.func @test_no_skip_block_erasure_linalg_map(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = linalg.map ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
                  outs(%0 : tensor<4xf32>)
    (%in0: f32, %in1: f32, %out: f32) {
      %2 = arith.addf %in0, %in1 : f32
      linalg.yield %2 : f32
    }
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: Block post-order erasures (no skip)
// CHECK:       Erasing block ^bb0 from region 0 from operation 'linalg.map'
// CHECK:       Erasing block ^bb0 from region 0 from operation 'func.func'
// CHECK:       Erasing block ^bb0 from region 0 from operation 'builtin.module'
