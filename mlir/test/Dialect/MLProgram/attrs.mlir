// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: #ml_program.extern : i32
"test.attributes"() {
  value = #ml_program.extern : i32
} : () -> ()

