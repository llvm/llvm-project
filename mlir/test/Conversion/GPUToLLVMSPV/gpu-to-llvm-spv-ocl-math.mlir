// RUN: mlir-opt %s -convert-gpu-to-llvm-spv | FileCheck %s -check-prefixes='CHECK,CHECK-NO-OCL'
// RUN: mlir-opt %s -convert-gpu-to-llvm-spv='convert-math-to-ocl=true' | FileCheck %s -check-prefixes='CHECK,CHECK-OCL'

gpu.module @kernels {
// CHECK-DAG:         llvm.func spir_funccc @_Z12get_local_idj(i32)
// CHECK-OCL-DAG:     llvm.func spir_funccc @_Z17__spirv_ocl_expm1f(f32)
// CHECK-NO-OCL-NOT:  llvm.func spir_funccc @_Z17__spirv_ocl_expm1f(f32)
// CHECK-LABEL: func.func @expm1_vector
  func.func @expm1_vector(%arg0: memref<32xvector<4xf32>>,
                          %arg1: memref<32xvector<4xf32>>) {
    // CHECK: llvm.call spir_funccc @_Z12get_local_idj
    %t_x = gpu.thread_id x
    // CHECK: %[[ARG0:.*]] = memref.load %arg0
    %v = memref.load %arg0[%t_x] : memref<32xvector<4xf32>>
    // CHECK-OCL: %[[EXT_0:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_0:.*]] = llvm.call spir_funccc @_Z17__spirv_ocl_expm1f(%[[EXT_0]])
    // CHECK-OCL: llvm.insertelement %[[VAL_0]]
    // CHECK-OCL: %[[EXT_1:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_1:.*]] = llvm.call spir_funccc @_Z17__spirv_ocl_expm1f(%[[EXT_1]])
    // CHECK-OCL: llvm.insertelement %[[VAL_1]]
    // CHECK-OCL: %[[EXT_2:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_2:.*]] = llvm.call spir_funccc @_Z17__spirv_ocl_expm1f(%[[EXT_2]])
    // CHECK-OCL: llvm.insertelement %[[VAL_2]]
    // CHECK-OCL: %[[EXT_3:.*]] = llvm.extractelement %[[ARG0]]
    // CHECK-OCL: %[[VAL_3:.*]] = llvm.call spir_funccc @_Z17__spirv_ocl_expm1f(%[[EXT_3]])
    // CHECK-OCL: %[[INS:.*]] = llvm.insertelement %[[VAL_3]]
    // CHECK-NO-OCL: %[[INS:.*]] = math.expm1 %[[ARG0]]
    %e = math.expm1 %v : vector<4xf32>
    // CHECK: memref.store %[[INS]], %arg1
    memref.store %e, %arg1[%t_x] : memref<32xvector<4xf32>>
    return
  }
}
