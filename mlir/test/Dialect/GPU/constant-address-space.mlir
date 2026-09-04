// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

gpu.module @test {
  // CHECK-LABEL: @constant_memref_basic
  // CHECK-SAME: (%{{.*}}: memref<16xf32, #gpu.address_space<constant>>)
  gpu.func @constant_memref_basic(%arg0: memref<16xf32, #gpu.address_space<constant>>) kernel {
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<16xf32, #gpu.address_space<constant>>
    gpu.return
  }

  // CHECK-LABEL: @constant_memref_multidim
  // CHECK: memref<4x8xf32, #gpu.address_space<constant>>
  gpu.func @constant_memref_multidim(%arg0: memref<4x8xf32, #gpu.address_space<constant>>) kernel {
    gpu.return
  }

  // CHECK-LABEL: @constant_memref_dynamic
  // CHECK: memref<?x?xf32, #gpu.address_space<constant>>
  gpu.func @constant_memref_dynamic(%arg0: memref<?x?xf32, #gpu.address_space<constant>>) kernel {
    gpu.return
  }
}
