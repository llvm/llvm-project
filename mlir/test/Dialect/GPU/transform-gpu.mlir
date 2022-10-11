// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file  -canonicalize -cse %s | FileCheck %s

!type = memref<2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-LABEL: func.func @saxpy2dblock(
// CHECK-SAME:    %[[ARGX:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGY:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGT:[0-9a-z]+]]: memref<32xf32>
func.func @saxpy2dblock(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %one = arith.constant 1 : index
//      CHECK:   gpu.launch
//      CHECK:   %[[BLKX:.*]] = gpu.block_id  x
//      CHECK:   %[[BLKY:.*]] = gpu.block_id  y
//      CHECK:   memref.load %[[ARGX]][%[[BLKX]], %[[BLKY]]]
//      CHECK:   memref.load %[[ARGY]][%[[BLKX]], %[[BLKY]]]
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  {thread_dim_mapping = [0, 1, 2]}
    gpu.terminator
  }
  return %y : !type
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0
    transform.gpu.map_foreach_to_blocks %funcop { blockDim = [12, 9, 1]}
  }
}

// -----

!type = memref<2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-LABEL: func.func @saxpy2d(
// CHECK-SAME:    %[[ARGX:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGY:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGT:[0-9a-z]+]]: memref<32xf32>
func.func @saxpy2d(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %one = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
//      CHECK:   %[[C1:.*]] = arith.constant 1 : index
//      CHECK:   %[[C12:.*]] = arith.constant 12 : index
//      CHECK:   %[[C9:.*]] = arith.constant 9 : index
//      CHECK:   %[[C7:.*]] = arith.constant 7 : index
//      CHECK:   gpu.launch async [%{{.*}}] blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C1]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C12]], %{{.*}} = %[[C9]], %{{.*}} = %[[C1]])
//      CHECK:   %[[TIDX:.*]] = gpu.thread_id  x
//      CHECK:   %[[TIDY:.*]] = gpu.thread_id  y
//      CHECK:   arith.cmpi ult, %[[TIDX]], %[[C9]] : index
//      CHECK:   arith.cmpi ult, %[[TIDY]], %[[C7]] : index
//      CHECK:   memref.load %[[ARGX]][%[[TIDY]], %[[TIDX]]]
//      CHECK:   memref.load %[[ARGY]][%[[TIDY]], %[[TIDX]]]
//      CHECK:   gpu.barrier
//      CHECK:   arith.cmpi ult, %[[TIDY]], %[[C1]] : index
//      CHECK:   memref.load %[[ARGT]][%[[TIDX]]]
//      CHECK:   gpu.barrier
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  {thread_dim_mapping = [1, 0, 2]}
     scf.foreach_thread (%i) in (%c12) {
        %7 = memref.load %t[%i] : !type1d
        %8 = arith.addf %alpha, %7 : f32
        memref.store %8, %t[%i] : !type1d
     }  {thread_dim_mapping = [0, 1, 2]}
    gpu.terminator
  }
  return %y : !type
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0
    transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [12, 9, 1] }
  }
}

// -----

!type4d = memref<32x64x4x32xf32>

// CHECK-LABEL: func.func @saxpy4d(
// CHECK-SAME:    %[[ARGX:[0-9a-z]+]]: memref<32x64x4x32xf32>
// CHECK-SAME:    %[[ARGY:[0-9a-z]+]]: memref<32x64x4x32xf32>
func.func @saxpy4d(%x: !type4d, %y: !type4d, %alpha : f32) -> !type4d {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
//      CHECK:   %[[C32:.*]] = arith.constant 32 : index
//      CHECK:   %[[C64:.*]] = arith.constant 64 : index
//      CHECK:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[C1:.*]] = arith.constant 1 : index
//      CHECK:   gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C32]], %{{.*}} = %[[C64]], %{{.*}} = %[[C1]]) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C32]], %{{.*}} = %[[C4]], %{{.*}} = %[[C1]])
//      CHECK:   %[[BLKX:.*]] = gpu.block_id  x
//      CHECK:   %[[BLKY:.*]] = gpu.block_id  y
//      CHECK:   %[[TIDX:.*]] = gpu.thread_id  x
//      CHECK:   %[[TIDY:.*]] = gpu.thread_id  y
//      CHECK:   memref.load %[[ARGX]][%[[BLKX]], %[[BLKY]], %[[TIDY]], %[[TIDX]]]
//      CHECK:   memref.load %[[ARGY]][%[[BLKX]], %[[BLKY]], %[[TIDY]], %[[TIDX]]]
  scf.foreach_thread (%i, %j) in (%c32, %c64) {
    scf.foreach_thread (%k, %l) in (%c4, %c32) {
      %4 = memref.load %x[%i, %j, %k, %l] : !type4d
      %5 = memref.load %y[%i, %j, %k, %l] : !type4d
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j, %k, %l] : !type4d
    }  {thread_dim_mapping = [1, 0, 2]}
  }  {thread_dim_mapping = [0, 1, 2]}
  return %y : !type4d
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %funcop = transform.structured.match ops{["func.func"]} in %arg0
    %gpuLaunch = transform.gpu.map_foreach_to_blocks %funcop { generate_gpu_launch }
    transform.gpu.map_nested_foreach_to_threads %gpuLaunch { blockDim = [32, 4, 1] }
  }
}

// -----

!type = memref<2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-LABEL: func.func @saxpy2d_no_barrier(
func.func @saxpy2d_no_barrier(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %one = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
//  CHECK-NOT:   gpu.barrier
//      CHECK:   return
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  {thread_dim_mapping = [1, 0, 2]}
    gpu.terminator
  }
  return %y : !type
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0
    transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [12, 9, 1], syncAfterDistribute = false }
  }
}
