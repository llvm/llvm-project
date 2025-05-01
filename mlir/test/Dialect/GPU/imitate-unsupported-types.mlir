// RUN: mlir-opt -verify-diagnostics -imitate-unsupported-types="source-types=bf16 target-types=i16" --canonicalize -split-input-file %s | FileCheck %s

// CHECK: module @builtin_module
module @builtin_module {
  // CHECK: gpu.module @gpu_func_module {
  gpu.module @gpu_func_module attributes{} {
    // CHECK-LABEL: gpu.func @arith_and_vector_ops
    // CHECK-SAME: (%[[ARG0:.*]]: memref<10x10xi16>, %[[ARG1:.*]]: memref<10x10xf32>, %[[ARG2:.*]]: vector<10x10xi16>, %[[ARG3:.*]]: memref<10x10xi16>, %[[ARG4:.*]]: vector<10x10xi16>) kernel
    gpu.func @arith_and_vector_ops(%arg0: memref<10x10xbf16>, %arg1: memref<10x10xf32>, %arg2: vector<10x10xbf16>, %arg3: memref<10x10xi16>, %arg4: vector<10x10xi16>) kernel attributes {} {

      %c0 = arith.constant 0 : index

      // CHECK: %[[ARG2_CAST:.*]] = arith.bitcast %[[ARG2]] : vector<10x10xi16> to vector<10x10xbf16>
      // CHECK: %[[LOAD1:.*]] = vector.load %[[ARG0]][%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>
      // CHECK: %[[BITCAST1:.*]] = arith.bitcast %[[LOAD1]] : vector<10x10xi16> to vector<10x10xbf16>
      %2 = vector.load %arg0[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>

      // CHECK: %[[ADDF:.*]] = arith.addf %[[BITCAST1]], %[[ARG2_CAST]] : vector<10x10xbf16>
      %add = arith.addf %2, %arg2 : vector<10x10xbf16>

      // CHECK: %[[EXTF1:.*]] = arith.extf %[[BITCAST1]] : vector<10x10xbf16> to vector<10x10xf32>
      %3 = arith.extf %2 : vector<10x10xbf16> to vector<10x10xf32>

      // CHECK: %[[EXTF2:.*]] = arith.extf %[[ADDF]] : vector<10x10xbf16> to vector<10x10xf32>
      %4 = arith.extf %add : vector<10x10xbf16> to vector<10x10xf32>

      // CHECK: %[[ADDF2:.*]] = arith.addf %[[EXTF1]], %[[EXTF2]] : vector<10x10xf32>
      %5 = arith.addf %3, %4 : vector<10x10xf32>

      // CHECK: %[[TRUNCF:.*]] = arith.truncf %[[ADDF2]] : vector<10x10xf32> to vector<10x10xbf16>
      %6 = arith.truncf %5 : vector<10x10xf32> to vector<10x10xbf16>

      // CHECK: %[[TRUNCF_CAST:.*]] = arith.bitcast %[[TRUNCF]] : vector<10x10xbf16> to vector<10x10xi16>
      // CHECK: vector.store %[[TRUNCF_CAST]], %[[ARG0]][%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>
      vector.store %6, %arg0[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>

      // CHECK: %[[LOAD2:.*]] = vector.load %[[ARG3]][%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>
      %7 = vector.load %arg3[%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>

      // CHECK: %[[ADDI:.*]] = arith.addi %[[LOAD2]], %[[ARG4]] : vector<10x10xi16>
      %8 = arith.addi %7, %arg4 : vector<10x10xi16>

      // CHECK: vector.store %[[ADDI]], %[[ARG3]][%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>
      vector.store %8, %arg3[%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>

      gpu.return
    }
  }
}

// -----


// CHECK: module @caller_callee_launch_func_module attributes {gpu.container_module}
module @caller_callee_launch_func_module attributes {gpu.container_module} {

  // CHECK: gpu.module @caller_callee_gpu_module {
  gpu.module @caller_callee_gpu_module attributes{} {

    // CHECK: gpu.func @caller_func(%[[ARG0:.*]]: memref<10x10xi16>, %[[ARG1:.*]]: vector<10x10xi16>) kernel {
    gpu.func @caller_func(%arg0: memref<10x10xbf16>, %arg1: vector<10x10xbf16>) kernel attributes {} {
      %c0 = arith.constant 0 : index

      // CHECK: %[[CALL_RET:.*]] = func.call @callee_constant_return() : () -> vector<10x10xi16>
      %func_result = func.call @callee_constant_return() : () -> vector<10x10xbf16>

      // CHECK: vector.store %[[CALL_RET]], %[[ARG0]][%c0, %c0] : memref<10x10xi16>, vector<10x10xi16>
      vector.store %func_result, %arg0[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>

      // CHECK: func.call @callee_func(%[[CALL_RET]]) : (vector<10x10xi16>) -> ()
      func.call @callee_func(%func_result) : (vector<10x10xbf16>) -> ()

      gpu.return
    }

    // CHECK: func.func @callee_constant_return() -> vector<10x10xi16> {
    func.func @callee_constant_return() -> vector<10x10xbf16> {
      // CHECK: arith.constant dense<16128> : vector<10x10xi16>
      %dense_const = arith.constant dense<5.000000e-01> : vector<10x10xbf16>
      func.return %dense_const : vector<10x10xbf16>
    }

    // CHECK: func.func @callee_func(%[[ARG:.*]]: vector<10x10xi16>) {
    func.func @callee_func(%arg0: vector<10x10xbf16>) {
      return
    }
  }

  // CHECK: func.func @gpu_launch_func(%[[ARG0:.*]]: memref<10x10xbf16>, %[[ARG1:.*]]: vector<10x10xbf16>) {
  func.func @gpu_launch_func(%arg0: memref<10x10xbf16>, %arg1: vector<10x10xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: arith.constant dense<16128> : vector<10x10xi16>
    %dense_const = arith.constant dense<5.000000e-01> : vector<10x10xbf16>
    // CHECK: arith.constant dense<6.015630e-01> : vector<10x10xbf16>
    %dense_const_2 = arith.constant dense<6.000000e-01> : vector<10x10xbf16>

    // CHECK: %[[ALLOC:.*]] = gpu.alloc () : memref<200xi8>
    %alloc = gpu.alloc () : memref<10x10xbf16>

    vector.store %dense_const_2, %alloc[%c0, %c0] : memref<10x10xbf16>, vector<10x10xbf16>
    // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]][%c0][] : memref<200xi8> to memref<10x10xi16>
    // CHECK: gpu.launch_func @caller_callee_gpu_module::@caller_func blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%[[VIEW]] : memref<10x10xi16>, %[[CST:.*]] : vector<10x10xi16>)
    gpu.launch_func @caller_callee_gpu_module::@caller_func
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%alloc: memref<10x10xbf16>, %dense_const: vector<10x10xbf16>)
    return
  }
}

// -----

// Only support alloc ops if it is in the same region as the launch op.
// Otherwise, it will leave an unresolved unrealized_conversion_cast in the IR
// due to typeconverter materialization.
module @unsupported_module attributes {gpu.container_module} {
  gpu.module @unsupported_gpu_module attributes{} {
    gpu.func @kernel(%arg0: memref<10x10xbf16>) kernel attributes {} {
      gpu.return
    }
  }

  func.func @gpu_launch_func(%arg0: memref<10x10xbf16>) {
    %c1 = arith.constant 1 : index
    // expected-error@+1 {{unresolved unrealized_conversion_cast left in IR after conversion}}
    gpu.launch_func @unsupported_gpu_module::@kernel
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0: memref<10x10xbf16>)
    return
  }

  func.func @main() {
    %alloc = memref.alloc () : memref<10x10xbf16>
    call @gpu_launch_func(%alloc) : (memref<10x10xbf16>) -> ()
    memref.dealloc %alloc : memref<10x10xbf16>
    return
  }
}

// -----

