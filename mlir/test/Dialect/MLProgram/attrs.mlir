// RUN: mlir-opt %s --allow-unregistered-dialect | mlir-opt --allow-unregistered-dialect | FileCheck %s

// CHECK: #ml_program.extern : i32
"unregistered.attributes"() {
  value = #ml_program.extern : i32
} : () -> ()

