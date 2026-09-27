// RUN: mlir-opt -convert-gpu-to-rocdl %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernel_module {
    // CHECK-LABEL: llvm.func @constant_load
    // CHECK-SAME: %{{.*}}: !llvm.ptr<4>
    gpu.func @constant_load(%arg0: memref<16xf32, #gpu.address_space<constant>>) kernel {
      %c0 = arith.constant 0 : index
      %v = memref.load %arg0[%c0] : memref<16xf32, #gpu.address_space<constant>>
      gpu.return
    }

    // CHECK-LABEL: llvm.func @constant_multidim
    // CHECK-SAME: %{{.*}}: !llvm.ptr<4>
    gpu.func @constant_multidim(%arg0: memref<4x8xf32, #gpu.address_space<constant>>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %v = memref.load %arg0[%c0, %c1] : memref<4x8xf32, #gpu.address_space<constant>>
      gpu.return
    }
  }
}
