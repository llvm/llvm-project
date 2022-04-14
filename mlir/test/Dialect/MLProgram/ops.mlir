// RUN: mlir-opt %s --allow-unregistered-dialect | mlir-opt --allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | mlir-opt --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: ml_program.func private @extern_func
ml_program.func private @extern_func(i32) -> i32

// CHECK-LABEL: ml_program.func @defined_func
ml_program.func @defined_func(%arg0 : i32) -> i32 {
  ml_program.return %arg0 : i32
}

// CHECK-LABEL: ml_program.subgraph private @extern_subgraph
ml_program.subgraph private @extern_subgraph(i32) -> i32

// CHECK-LABEL: ml_program.subgraph @compute_subgraph
ml_program.subgraph @compute_subgraph(%arg0 : i32) -> i32 {
  %1 = "unregistered.dummy"(%0) : (i32) -> i32
  %0 = "unregistered.dummy"(%arg0) : (i32) -> i32
  ml_program.output %0 : i32
}
