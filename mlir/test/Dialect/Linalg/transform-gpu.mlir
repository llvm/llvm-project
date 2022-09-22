// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file %s | FileCheck %s

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
//      CHECK:   gpu.launch
//      CHECK:   %[[TIDX:.*]] = gpu.thread_id  x
//      CHECK:   %[[TIDY:.*]] = gpu.thread_id  y
//      CHECK:   %[[C9:.*]] = arith.constant 9 : index
//      CHECK:   arith.cmpi ult, %[[TIDX]], %[[C9]] : index
//      CHECK:   %[[C7:.*]] = arith.constant 7 : index
//      CHECK:   arith.cmpi ult, %[[TIDY]], %[[C7]] : index
//      CHECK:   memref.load %[[ARGX]][%[[TIDY]], %[[TIDX]]]
//      CHECK:   memref.load %[[ARGY]][%[[TIDY]], %[[TIDX]]]
//      CHECK:   gpu.barrier
//      CHECK:   %[[TIDX2:.*]] = gpu.thread_id  x
//      CHECK:   %[[TIDY2:.*]] = gpu.thread_id  y
//      CHECK:   %[[C1:.*]] = arith.constant 1 : index
//      CHECK:   arith.cmpi ult, %[[TIDY2]], %[[C1]] : index
//      CHECK:   memref.load %[[ARGT]][%[[TIDX2]]]
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
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0
    transform.structured.map_nested_foreach_thread_to_gpu_threads %funcop { blockDim = [12, 9, 1] }
  }
}

