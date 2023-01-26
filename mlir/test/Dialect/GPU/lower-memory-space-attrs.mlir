// RUN: mlir-opt %s -split-input-file -gpu-lower-memory-space-attributes | FileCheck %s
// RUN: mlir-opt %s -split-input-file -gpu-lower-memory-space-attributes="private=0 global=0" | FileCheck %s --check-prefix=CUDA

gpu.module @kernel {
  gpu.func @private(%arg0: f32) private(%arg1: memref<4xf32, #gpu.address_space<private>>) {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<4xf32, #gpu.address_space<private>>
    gpu.return
  }
}

//      CHECK:  gpu.func @private
// CHECK-SAME:    private(%{{.+}}: memref<4xf32, 5>)
//      CHECK:  memref.store
// CHECK-SAME:   : memref<4xf32, 5>

//      CUDA:  gpu.func @private
// CUDA-SAME:    private(%{{.+}}: memref<4xf32>)
//      CUDA:  memref.store
// CUDA-SAME:   : memref<4xf32>

// -----

gpu.module @kernel {
  gpu.func @workgroup(%arg0: f32) workgroup(%arg1: memref<4xf32, #gpu.address_space<workgroup>>) {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<4xf32, #gpu.address_space<workgroup>>
    gpu.return
  }
}

//      CHECK:  gpu.func @workgroup
// CHECK-SAME:    workgroup(%{{.+}}: memref<4xf32, 3>)
//      CHECK:  memref.store
// CHECK-SAME:   : memref<4xf32, 3>

// -----

gpu.module @kernel {
  gpu.func @nested_memref(%arg0: memref<4xmemref<4xf32, #gpu.address_space<global>>, #gpu.address_space<global>>) {
    %c0 = arith.constant 0 : index
    memref.load %arg0[%c0] : memref<4xmemref<4xf32, #gpu.address_space<global>>, #gpu.address_space<global>>
    gpu.return
  }
}

//      CHECK:  gpu.func @nested_memref
// CHECK-SAME:    (%{{.+}}: memref<4xmemref<4xf32, 1>, 1>)
//      CHECK:  memref.load
// CHECK-SAME:   : memref<4xmemref<4xf32, 1>, 1>

//      CUDA:  gpu.func @nested_memref
// CUDA-SAME:    (%{{.+}}: memref<4xmemref<4xf32>>)
//      CUDA:  memref.load
// CUDA-SAME:   : memref<4xmemref<4xf32>>
