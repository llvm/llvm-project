// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  // func.func @foo(%size : index) -> memref<?xf32> {
  //   %t0 = gpu.wait async
  //   %t1 = gpu.wait async [%t0]
  //   %0 = gpu.alloc [%t0, %t1] (%size) : memref<?xf32>
  //   // gpu.wait [%1]
  //   return %0 : memref<?xf32>
  // }

  gpu.module @foo {
    gpu.func @bar() kernel {
      gpu.return
    }
  }

  func.func @main() {
    %c1 = arith.constant 1 : index

    %t0 = gpu.wait async
    %t1 = gpu.wait async [%t0]
    %token = gpu.launch_func async [%t0, %t1] @foo::@bar
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
    gpu.wait [%token]
    return
  }

}
