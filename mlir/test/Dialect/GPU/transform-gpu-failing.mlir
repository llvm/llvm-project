// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file  -canonicalize -cse --verify-diagnostics %s

func.func @map_nested_forall_to_threads_not_gpu_launch() -> () {
  %1 = tensor.empty() : tensor<4xf32>
  return
}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{Given target is not a gpu.launch}}
  %1 = transform.gpu.map_nested_forall_to_threads %funcop block_dims = [1, 1, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @map_nested_forall_to_threads_excessive_threads(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c900) {
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
    scf.forall (%i, %j) in (%c7, %c9) {
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
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{Trying to launch a GPU kernel with grid_dims = (1, 1, 1) block_dims = (1200, 9, 1). It is larger than the limits.}}
  // expected-note @below {{"block_dims" is too large}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [1200, 9, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @map_nested_forall_to_threads_fewer_threads(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c900) {
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
    scf.forall (%i, %j) in (%c7, %c9) {
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
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{the number of required parallel resources (blocks or threads) 6300 overflows the number of available resources 512}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [128, 4, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @map_nested_forall_to_threads_dynamic_trip_count(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token, %c9 : index, %c7 : index) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c900) {
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
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{requires statically sized, normalized forall op}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [128, 4, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @map_nested_forall_to_threads_not_buffer(%x: tensor<32x32xf32>, %y: tensor<32x32xf32>, %z: tensor<32x32xf32>, %stream : !gpu.async.token) {
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
^bb1(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %forall, %tiled = transform.structured.tile_to_forall_op %matmul num_threads [2, 3, 1] (mapping = [ #gpu.thread<y>, #gpu.thread<x>, #gpu.thread<z> ] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{only bufferized scf.forall can be mapped}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [96, 4, 1] : (!transform.any_op) -> !transform.any_op
}

// -----


func.func @map_forall_to_blocks_not_gpu_launch() -> () {
  // expected-note @below {{when applied to this payload op}}
  %1 = tensor.empty() : tensor<4xf32>
  return
}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{Given target is not gpu.launch}}
  %1 = transform.gpu.map_forall_to_blocks %funcop : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @map_forall_to_blocks_not_unique(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  // expected-note @below {{when applied to this payload op}}
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c900) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }

    scf.forall (%i, %j) in (%c7, %c9) {
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
^bb0(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{could not find a unique topLevel scf.forall}}
  %1 = transform.gpu.map_forall_to_blocks %funcop : (!transform.any_op) -> !transform.any_op
}

// -----

// expected-note @below {{when applied to this payload op}}
func.func @map_forall_to_blocks_large_loop(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c65537 = arith.constant 65536 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index

  scf.forall (%i, %j) in (%c7, %c65537) {
      %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
      %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
  }  { mapping = [#gpu.thread<x>, #gpu.thread<y>] }

  scf.forall (%i, %j) in (%c7, %c9) {
      %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
      %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
  }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }

  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{could not find a unique topLevel scf.forall}}
  %1 = transform.gpu.map_forall_to_blocks %funcop { generate_gpu_launch } : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @map_forall_to_blocks_large_loop(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c65535 = arith.constant 65535 : index
  scf.forall (%i, %j) in (%c65535, %c65535) {
      %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
      %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
  }  { mapping = [#gpu.block<x>, #gpu.block<y>] }
  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{Trying to launch a GPU kernel with grid_dims = (65535, 65535, 1) block_dims = (1, 1, 1). It is larger than the limits.}}
  %1 = transform.gpu.map_forall_to_blocks %funcop generate_gpu_launch : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<32x32xf32>
func.func @saxpy2d_singleloop(%x: !type, %y: !type, %stream : !gpu.async.token) -> !type {
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c32, %c32) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = arith.mulf %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  { mapping = [#gpu.thread<x>, #gpu.warp<y>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{cannot mix different mapping types, use nesting}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<32x32xf32>
func.func @saxpy2d_singleloop(%x: !type, %y: !type, %stream : !gpu.async.token) -> !type {
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c32, %c32) {
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
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{duplicate attribute, cannot map different loops to the same mapping id}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<32x32xf32>
func.func @saxpy2d_singleloop(%x: !type, %y: !type, %stream : !gpu.async.token) -> !type {
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c32, %c32) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = arith.mulf %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  { mapping = [#gpu.thread<x>, #gpu.thread<linear_dim_0>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{cannot mix linear and non-linear mapping modes}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

// expected-note @below {{when applied to this payload op}}
module {
transform.sequence failures(propagate) {
^bb1(%op: !transform.any_op):
  // expected-error @below {{could not find a unique topLevel scf.forall}}
  %gpu_launch = transform.gpu.map_forall_to_blocks %op generate_gpu_launch grid_dims = [1, 1, 1]
    : (!transform.any_op) -> !transform.any_op
}
}

// -----

func.func public @improperly_sized_grid_dims(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
  scf.forall (%arg3, %arg4) in (1, 1) {
    linalg.matmul ins(%arg0, %arg1 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg2 : memref<32x32xf32>)
  } {mapping = [#gpu.block<x>, #gpu.block<y>]}
  return
}

transform.sequence  failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %arg0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{transform requires empty or size-3 grid_dims}}
  %5 = transform.gpu.map_forall_to_blocks %arg1 generate_gpu_launch grid_dims = [50, 16] : (!transform.any_op) -> !transform.any_op
}

// -----

func.func public @missing_mapping_attribute(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
  scf.forall (%arg3, %arg4) in (1, 1) {
    linalg.matmul ins(%arg0, %arg1 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg2 : memref<32x32xf32>)
  }
  return
}

transform.sequence  failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %arg0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{scf.forall op requires a mapping attribute}}
  %5 = transform.gpu.map_forall_to_blocks %arg1 generate_gpu_launch grid_dims = [50, 16, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

func.func public @not_a_block_mapping_attribute(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
  scf.forall (%arg3, %arg4) in (1, 1) {
    linalg.matmul ins(%arg0, %arg1 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg2 : memref<32x32xf32>)
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
  return
}

transform.sequence  failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %arg0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{scf.forall op requires a mapping attribute of kind 'block'}}
  %5 = transform.gpu.map_forall_to_blocks %arg1 generate_gpu_launch grid_dims = [50, 16, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @not_a_thread_or_warp_mapping_attribute(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c900) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.block<y>, #gpu.block<x>] }
    gpu.terminator
  }

  return %y : memref<2 x 32 x f32>
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{scf.forall op requires a mapping attribute of kind 'thread' or 'warp'}}
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [1, 1, 1] : (!transform.any_op) -> !transform.any_op
}
