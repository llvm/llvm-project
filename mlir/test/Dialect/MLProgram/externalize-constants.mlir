// RUN: mlir-opt %s --mlprogram-externalize-constants="min-size=2 output-filename=%t.mlirbc" | FileCheck %s --check-prefix=EXT
// RUN: mlir-opt %s --mlprogram-externalize-constants="min-size=2 output-filename=%t.mlirbc" | mlir-opt --mlprogram-repopulate-constants="input-filename=%t.mlirbc" | FileCheck %s --check-prefix=REP
// RUN: mlir-opt %s --mlprogram-externalize-constants="min-size=2 drop-locations=true" -mlir-print-debuginfo | FileCheck %s --check-prefix=LOC

// EXT-DAG: ml_program.global public @_weight_0
// EXT-DAG: #ml_program.extern
// EXT-DAG: ml_program.global public @_weight_1
// EXT-DAG: #ml_program.extern

// EXT-LABEL: func.func @main
// EXT: %[[LOAD0:.*]] = ml_program.global_load_const @_weight_0 : tensor<4xf32>
// EXT: %[[LOAD1:.*]] = ml_program.global_load_const @_weight_1 : tensor<2xf32>
// EXT: return %[[LOAD0]], %[[LOAD1]]

// REP-LABEL: func.func @main
// REP: %[[CST0:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
// REP: %[[CST1:.*]] = arith.constant dense<[5.000000e+00, 6.000000e+00]> : tensor<2xf32>
// REP: return %[[CST0]], %[[CST1]]

// LOC-DAG: ml_program.global public @_weight_0{{.*}}loc(#loc1)
// LOC-DAG: ml_program.global public @_weight_1{{.*}}loc(#loc1)
// LOC-LABEL: func.func @main
// LOC: %[[LOAD0:.*]] = ml_program.global_load_const @_weight_0 : tensor<4xf32> loc(#loc1)
// LOC: %[[LOAD1:.*]] = ml_program.global_load_const @_weight_1 : tensor<2xf32> loc(#loc1)
// LOC: #loc1 = loc(unknown)

func.func @main() -> (tensor<4xf32>, tensor<2xf32>) {
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32> loc("loc1")
  %cst2 = arith.constant dense<[5.0, 6.0]> : tensor<2xf32> loc("loc2")
  %cst3 = arith.constant dense<[7.0]> : tensor<1xf32> loc("loc3")
  return %cst, %cst2 : tensor<4xf32>, tensor<2xf32>
}
