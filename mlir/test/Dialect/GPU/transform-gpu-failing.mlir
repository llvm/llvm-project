// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file  -canonicalize -cse --verify-diagnostics %s

func.func @map_nested_foreach_to_threads_not_gpu_launch() -> () {
  %1 = tensor.empty() : tensor<4xf32>
  return
}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{Given target is not gpu.launch}}
  %1 = transform.gpu.map_nested_foreach_to_threads %funcop
}

// -----

func.func @map_nested_foreach_to_threads_excessive_threads(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c900) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }

  %name2 = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }

  return %y : memref<2 x 32 x f32>
}
transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{Trying to launch a GPU kernel with gridDim = (1, 1, 1) blockDim = (1200, 9, 1). It is larger than the limits.}}
  // expected-note @below {{"blockDim" is very large}}
  transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [1200, 9, 1] }
}

// -----

func.func @map_nested_foreach_to_threads_fewer_threads(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c900) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }

  %name2 = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }

  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{The requested GPU threads are fewer than the number of loop trip counts. Try to tile scf.foreach_thread before mapping or set small blockDim.}}
  transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [128, 4, 1] }
}

// -----

func.func @map_nested_foreach_to_threads_dynamic_trip_count(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token, %c9 : index, %c7 : index) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c900) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }
  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{unsupported dynamic blockdim size}}
  transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [128, 4, 1] }
}

// -----

func.func @map_nested_foreach_to_threads_not_buffer(%x: tensor<32x32xf32>, %y: tensor<32x32xf32>, %z: tensor<32x32xf32>, %stream : !gpu.async.token) {
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    %t = linalg.matmul ins(%x, %y: tensor<32x32xf32>, tensor<32x32xf32>) outs(%z : tensor<32x32xf32>) -> tensor<32x32xf32>
    gpu.terminator
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %foreach, %tiled = transform.structured.tile_to_foreach_thread_op %matmul num_threads [10, 20, 30] (mapping = [ #gpu.thread<y>, #gpu.thread<x>, #gpu.thread<z> ] )
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{only bufferized scf.foreach_thread lowers to gpu.thread_id}}
  transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [128, 4, 1] }
}

// -----


func.func @map_foreach_to_blocks_not_gpu_launch() -> () {
  // expected-note @below {{when applied to this payload op}}
  %1 = tensor.empty() : tensor<4xf32>
  return
}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{Given target is not gpu.launch}}
  %1 = transform.gpu.map_foreach_to_blocks %funcop
}

// -----

func.func @map_foreach_to_blocks_not_unique(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  // expected-note @below {{when applied to this payload op}}
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c7, %c900) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }

    scf.foreach_thread (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }

  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{could not find a unique topLevel scf.foreach_thread}}
  %1 = transform.gpu.map_foreach_to_blocks %funcop
}

// -----

// expected-note @below {{when applied to this payload op}}
func.func @map_foreach_to_blocks_large_loop(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c65537 = arith.constant 65536 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index

  scf.foreach_thread (%i, %j) in (%c7, %c65537) {
      %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
      %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
  }  { mapping = [#gpu.thread<x>, #gpu.thread<y>] }

  scf.foreach_thread (%i, %j) in (%c7, %c9) {
      %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
      %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
  }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }

  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{could not find a unique topLevel scf.foreach_thread}}
  %1 = transform.gpu.map_foreach_to_blocks %funcop { generate_gpu_launch }
}

// -----

func.func @map_foreach_to_blocks_large_loop(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c65535 = arith.constant 65535 : index
  scf.foreach_thread (%i, %j) in (%c65535, %c65535) {
      %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
      %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
  }  { mapping = [#gpu.block<x>, #gpu.block<y>] }
  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{Trying to launch a GPU kernel with gridDim = (65535, 65535, 1) blockDim = (1, 1, 1). It is larger than the limits.}}
  %1 = transform.gpu.map_foreach_to_blocks %funcop { generate_gpu_launch }
}

// -----

!type = memref<32x32xf32>
func.func @saxpy2d_singleloop(%x: !type, %y: !type, %stream : !gpu.async.token) -> !type {
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c32, %c32) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = arith.mulf %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  { mapping = [#gpu.thread<x>, #gpu.thread<x>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{#gpu.thread<x> is duplicated, cannot map different loops to the same processor}}
  transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [32, 32]}
}

// -----

!type = memref<32x32xf32>
func.func @saxpy2d_wrong_mapping(%x: !type, %y: !type, %stream : !gpu.async.token) -> !type {
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.foreach_thread (%i, %j) in (%c32, %c32) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = arith.mulf %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  { mapping = [#gpu.block<x>, #gpu.block<x>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{mapping must be one of #gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>}}
  transform.gpu.map_nested_foreach_to_threads %funcop { blockDim = [32, 32]}
}

