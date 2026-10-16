// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// CHECK: module attributes {dlti = #dlti.map<"module" = 1 : i32>, gpu.container_module}
module attributes {dlti = #dlti.map<"module" = 1 : i32>, gpu.container_module} {
  // CHECK: func.func @new() attributes {dlti = #dlti.map<"func" = 2 : i32>}
  func.func @new() attributes {dlti = #dlti.map<"func" = 2 : i32>} {
    return
  }
  // CHECK: gpu.module @new_gpu dlti = #dlti.map<"gpu" = 3 : i32>
  gpu.module @new_gpu dlti = #dlti.map<"gpu" = 3 : i32> {
  }
}
