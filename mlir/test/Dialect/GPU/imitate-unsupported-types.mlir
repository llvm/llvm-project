// RUN: mlir-opt -verify-diagnostics -imitate-unsupported-types="source-types=bf16 target-types=i16" --canonicalize -split-input-file %s | FileCheck %s

// CHECK: module @builtin_module
module @builtin_module {
  // CHECK: gpu.module @gpu_func_module
  gpu.module @gpu_func_module {
    // CHECK: gpu.func @arith_and_vector_ops
    // CHECK-SAME: (%[[ARG0:.*]]: memref<10x10xi16>, %[[ARG1:.*]]: memref<10x10xf32>, %[[ARG2:.*]]: vector<10x10xi16>, %[[ARG3:.*]]: memref<10x10xi16>, %[[ARG4:.*]]: vector<10x10xi16>) kernel
    gpu.func @arith_and_vector_ops(
        %arg0: memref<10x10xbf16>,
        %arg1: memref<10x10xf32>,
        %arg2: vector<10x10xbf16>,
        %arg3: memref<10x10xi16>,
        %arg4: vector<10x10xi16>
    ) kernel {
      // CHECK: %[[C0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index

      // CHECK: %[[CAST_ARG2:.*]] = arith.bitcast %[[ARG2]] : vector<10x10xi16> to vector<10x10xbf16>
      // CHECK: %[[LOAD_ARG0:.*]] = vector.load %[[ARG0]][%[[C0]], %[[C0]]] : memref<10x10xi16>, vector<10x10xi16>
      // CHECK: %[[CAST_LOAD:.*]] = arith.bitcast %[[LOAD_ARG0]] : vector<10x10xi16> to vector<10x10xbf16>
      %0 = vector.load %arg0[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>

      // CHECK: %[[ADDF:.*]] = arith.addf %[[CAST_LOAD]], %[[CAST_ARG2]] : vector<10x10xbf16>
      %1 = arith.addf %0, %arg2 : vector<10x10xbf16>

      // CHECK: %[[EXT0:.*]] = arith.extf %[[CAST_LOAD]] : vector<10x10xbf16> to vector<10x10xf32>
      %2 = arith.extf %0 : vector<10x10xbf16> to vector<10x10xf32>

      // CHECK: %[[EXT1:.*]] = arith.extf %[[ADDF]] : vector<10x10xbf16> to vector<10x10xf32>
      %3 = arith.extf %1 : vector<10x10xbf16> to vector<10x10xf32>

      // CHECK: %[[FADD:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : vector<10x10xf32>
      %4 = arith.addf %2, %3 : vector<10x10xf32>

      // CHECK: %[[TRUNC:.*]] = arith.truncf %[[FADD]] : vector<10x10xf32> to vector<10x10xbf16>
      %5 = arith.truncf %4 : vector<10x10xf32> to vector<10x10xbf16>

      // CHECK: %[[CAST_TRUNC:.*]] = arith.bitcast %[[TRUNC]] : vector<10x10xbf16> to vector<10x10xi16>
      // CHECK: vector.store %[[CAST_TRUNC]], %[[ARG0]][%[[C0]], %[[C0]]] : memref<10x10xi16>, vector<10x10xi16>
      vector.store %5, %arg0[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>

      // CHECK: %[[LOAD2:.*]] = vector.load %[[ARG3]][%[[C0]], %[[C0]]] : memref<10x10xi16>, vector<10x10xi16>
      %6 = vector.load %arg3[%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>

      // CHECK: %[[ADDI:.*]] = arith.addi %[[LOAD2]], %[[ARG4]] : vector<10x10xi16>
      %7 = arith.addi %6, %arg4 : vector<10x10xi16>

      // CHECK: vector.store %[[ADDI]], %[[ARG3]][%[[C0]], %[[C0]]] : memref<10x10xi16>, vector<10x10xi16>
      vector.store %7, %arg3[%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>

      // CHECK: gpu.return
      gpu.return
    }
  }
}

// -----


// CHECK: module @caller_callee_launch_func_module attributes {gpu.container_module}
module @caller_callee_launch_func_module attributes {gpu.container_module} {
  // CHECK: gpu.module @caller_callee_gpu_module {
  gpu.module @caller_callee_gpu_module attributes{} {
    // CHECK: gpu.func @caller_func
    // CHECK-SAME: (%[[ARG0:.*]]: memref<10x10xi16>, %[[ARG1:.*]]: vector<10x10xi16>) kernel
    gpu.func @caller_func(%arg0: memref<10x10xbf16>, %arg1: vector<10x10xbf16>) kernel attributes {} {
      // CHECK: %[[C0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index

      // CHECK: %[[RET:.*]] = func.call @callee_constant_return() : () -> vector<10x10xi16>
      %func_result = func.call @callee_constant_return() : () -> vector<10x10xbf16>

      // CHECK: vector.store %[[RET]], %[[ARG0]][%[[C0]], %[[C0]]] : memref<10x10xi16>, vector<10x10xi16>
      vector.store %func_result, %arg0[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>

      // CHECK: func.call @callee_func(%[[RET]]) : (vector<10x10xi16>) -> ()
      func.call @callee_func(%func_result) : (vector<10x10xbf16>) -> ()

      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: func.func @callee_constant_return() -> vector<10x10xi16> {
    func.func @callee_constant_return() -> vector<10x10xbf16> {
      // CHECK: %[[CST:.*]] = arith.constant dense<16128> : vector<10x10xi16>
      %dense_const = arith.constant dense<5.000000e-01> : vector<10x10xbf16>
      // CHECK: return %[[CST]] : vector<10x10xi16>
      func.return %dense_const : vector<10x10xbf16>
    }

    // CHECK: func.func @callee_func(%[[ARG:.*]]: vector<10x10xi16>) {
    func.func @callee_func(%arg0: vector<10x10xbf16>) {
      return
    }
  }

  // CHECK: func.func @gpu_launch_func(
  func.func @gpu_launch_func(%arg0: memref<10x10xbf16>, %arg1: vector<10x10xbf16>) {

    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Handling bf16 constants, dealing with constants for both cases:
    //  - not used in gpu.launch_func (no conversion)
    //  - used in gpu.launch_func (needs conversion to i16)

    // CHECK: %[[BF16_CONST:.*]] = arith.constant dense<5.000000e-01> : vector<10x10xbf16>
    // CHECK: %[[I16_CONST:.*]] = arith.constant dense<16128> : vector<10x10xi16>
    %dense_const = arith.constant dense<5.000000e-01> : vector<10x10xbf16>

    // CHECK: %[[BF16_CONST_2:.*]] = arith.constant dense<1.500000e+00> : vector<10x10xbf16>
    %dense_const_2 = arith.constant dense<1.500000e+00> : vector<10x10xbf16>

    // CHECK: %[[ADDF:.*]] = arith.addf %arg1, %[[BF16_CONST]] : vector<10x10xbf16>
    %add = arith.addf %dense_const, %arg1 : vector<10x10xbf16>

    // CHECK: vector.store %[[ADDF]], %arg0[%[[C0]], %[[C0]]] : memref<10x10xbf16>, vector<10x10xbf16>
    vector.store %add, %arg0[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>

    // CHECK: %[[ALLOC:.*]] = gpu.alloc () : memref<10x10xbf16>
    %alloc = gpu.alloc () : memref<10x10xbf16>
    // CHECK: %[[BITCAST:.*]] = arith.bitcast %[[ALLOC]] : memref<10x10xbf16> to memref<10x10xi16>
    // CHECK: vector.store %[[BF16_CONST_2]], %[[ALLOC]][%[[C0]], %[[C0]]] : memref<10x10xbf16>, vector<10x10xbf16>
    vector.store %dense_const_2, %alloc[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>


    // CHECK: gpu.launch_func @caller_callee_gpu_module::@caller_func
    // CHECK-SAME: args(%[[BITCAST]] : memref<10x10xi16>, %[[I16_CONST]] : vector<10x10xi16>)
    gpu.launch_func @caller_callee_gpu_module::@caller_func
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%alloc: memref<10x10xbf16>, %dense_const: vector<10x10xbf16>)
    return
  }
}

// -----


// CHECK: #map = affine_map<(d0, d1) -> (d1, d0)>
#map = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: module @module_multi_level_call attributes {gpu.container_module} {
module @module_multi_level_call attributes {gpu.container_module} {
  // CHECK: gpu.module @gpu_module_multi_level_call {
  gpu.module @gpu_module_multi_level_call {
    // CHECK: gpu.func @kernel(%[[K_ARG:.*]]: memref<10x10xi16>) kernel {
    gpu.func @kernel(%arg0: memref<10x10xi16>) kernel {
      // CHECK: gpu.return
      gpu.return
    }

    // CHECK: gpu.func @affine_memref_arg(%[[AFF_ARG:.*]]: memref<100x100xi16, #map, 2>) kernel {
    gpu.func @affine_memref_arg(%arg0: memref<100x100xi16, #map, 2>) kernel {
      // CHECK: gpu.return
      gpu.return
    }
  }

  // CHECK-LABEL: func.func @gpu_launch_func
  func.func @gpu_launch_func(%arg0: memref<10x10xbf16>, %arg1: memref<100x100xbf16, #map, 2>) {
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    %c1 = arith.constant 1 : index

    // CHECK: %[[AFF_CAST:.*]] = arith.bitcast %[[ARG1:.*]] : memref<100x100xbf16, #map, 2> to memref<100x100xi16, #map, 2>
    %0 = arith.bitcast %arg1 : memref<100x100xbf16, #map, 2> to memref<100x100xi16, #map, 2>

    // CHECK: %[[BF16_CAST:.*]] = arith.bitcast %[[ARG0:.*]] : memref<10x10xbf16> to memref<10x10xi16>
    %1 = arith.bitcast %arg0 : memref<10x10xbf16> to memref<10x10xi16>

    // CHECK: gpu.launch_func @gpu_module_multi_level_call::@kernel
    // CHECK-SAME: args(%[[BF16_CAST]] : memref<10x10xi16>)
    gpu.launch_func @gpu_module_multi_level_call::@kernel
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%1 : memref<10x10xi16>)

    // CHECK: gpu.launch_func @gpu_module_multi_level_call::@affine_memref_arg
    // CHECK-SAME: args(%[[AFF_CAST]] : memref<100x100xi16, #map, 2>)
    gpu.launch_func @gpu_module_multi_level_call::@affine_memref_arg
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%0 : memref<100x100xi16, #map, 2>)
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @main
  func.func @main() {
    // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<10x10xbf16>
    %alloc = memref.alloc() : memref<10x10xbf16>
    // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<100x100xbf16, #map, 2>
    %alloc_0 = memref.alloc() : memref<100x100xbf16, #map, 2>
    // CHECK: call @gpu_launch_func(%[[ALLOC0]], %[[ALLOC1]])
    call @gpu_launch_func(%alloc, %alloc_0) : (memref<10x10xbf16>, memref<100x100xbf16, #map, 2>) -> ()
    // CHECK: memref.dealloc %[[ALLOC0]]
    memref.dealloc %alloc : memref<10x10xbf16>
    // CHECK: memref.dealloc %[[ALLOC1]]
    memref.dealloc %alloc_0 : memref<100x100xbf16, #map, 2>
    // CHECK: return
    return
  }
}



