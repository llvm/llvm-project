// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func @not_enough_sizes(%sz : index) {
  // expected-error@+1 {{expected 6 operands, but found 5}}
  "gpu.launch"(%sz, %sz, %sz, %sz, %sz) ({
    gpu.return
  }) : (index, index, index, index, index) -> ()
  return
}

// -----

func @no_region_attrs(%sz : index) {
  // expected-error@+1 {{unexpected number of region arguments}}
 "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) ({
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index):
    gpu.return
  }) : (index, index, index, index, index, index) -> ()
  return
}

// -----

func @launch_requires_gpu_return(%sz : index) {
  // @expected-note@+1 {{in 'gpu.launch' body region}}
  gpu.launch blocks(%bx, %by, %bz) in (%sbx = %sz, %sby = %sz, %sbz = %sz)
             threads(%tx, %ty, %tz) in (%stx = %sz, %sty = %sz, %stz = %sz) {
    // @expected-error@+1 {{expected 'gpu.terminator' or a terminator with successors}}
    return
  }
  return
}

// -----

func @launch_func_too_few_operands(%sz : index) {
  // expected-error@+1 {{expected 6 or more operands}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz)
      : (index, index, index, index, index) -> ()
  return
}

// -----

func @launch_func_missing_parent_module_attribute(%sz : index) {
  // expected-error@+1 {{expected the closest surrounding module to have the 'gpu.container_module' attribute}}
  "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz) {foo = "bar"}
      : (index, index, index, index, index, index) -> ()
  return
}

// -----

module attributes {gpu.container_module} {
  func @launch_func_missing_callee_attribute(%sz : index) {
    // expected-error@+1 {{string attribute 'kernel' must be specified}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz) {foo = "bar"}
        : (index, index, index, index, index, index) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  func @launch_func_missing_module_attribute(%sz : index) {
    // expected-error@+1 {{attribute 'kernel_module' must be specified}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz) {kernel = "launch_func_missing_kernel_attr"}
        : (index, index, index, index, index, index) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  func @launch_func_no_function_attribute(%sz : index) {
    // expected-error@+1 {{string attribute 'kernel' must be specified}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz) {kernel = 10}
        : (index, index, index, index, index, index) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  func @launch_func_module_attribute_wrong_type(%sz : index) {
    // expected-error@+1 {{symbol reference attribute 'kernel_module' must be specified}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz)
    {kernel = "launch_func_module_attribute_wrong_type", kernel_module = 10}
        : (index, index, index, index, index, index) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  func @launch_func_undefined_module(%sz : index) {
    // expected-error@+1 {{kernel module 'kernels' is undefined}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz)
    { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  module @kernels {
  }

  func @launch_func_missing_module_attribute(%sz : index) {
    // expected-error@+1 {{kernel module 'kernels' is undefined}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz)
    { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels { }

  func @launch_func_undefined_function(%sz : index) {
    // expected-error@+1 {{kernel function 'kernel_1' is undefined}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz)
    { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_1(%arg1 : !llvm<"float*">) kernel {
      gpu.return
    }
  }

  func @launch_func_missing_kernel_attr(%sz : index, %arg : !llvm<"float*">) {
    // xpected-error@+1 {{kernel function is missing the 'gpu.kernel' attribute}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz, %arg)
    {kernel = "kernel_1", kernel_module = @kernels}
        : (index, index, index, index, index, index, !llvm<"float*">) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_1(%arg1 : !llvm<"float*">) attributes { gpu.kernel } {
      gpu.return
    }
  }

  func @launch_func_kernel_operand_size(%sz : index, %arg : !llvm<"float*">) {
    // expected-error@+1 {{got 2 kernel operands but expected 1}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz, %arg, %arg)
        {kernel = "kernel_1", kernel_module = @kernels}
        : (index, index, index, index, index, index, !llvm<"float*">,
           !llvm<"float*">) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_1(%arg1 : f32) attributes { gpu.kernel } {
      gpu.return
    }
  }

  func @launch_func_kernel_operand_types(%sz : index, %arg : f32) {
    // expected-err@+1 {{type of function argument 0 does not match}}
    "gpu.launch_func"(%sz, %sz, %sz, %sz, %sz, %sz, %arg)
        {kernel = "kernel_1", kernel_module = @kernels}
        : (index, index, index, index, index, index, f32) -> ()
    return
  }
}

// -----

func @illegal_dimension() {
  // expected-error@+1 {{dimension "o" is invalid}}
  %tIdX = "gpu.thread_id"() {dimension = "o"} : () -> (index)

  return
}

// -----

func @illegal_dimension() {
  // expected-error@+1 {{dimension "o" is invalid}}
  %bDimX = "gpu.block_dim"() {dimension = "o"} : () -> (index)

  return
}

// -----

func @illegal_dimension() {
  // expected-error@+1 {{dimension "o" is invalid}}
  %bIdX = "gpu.block_id"() {dimension = "o"} : () -> (index)

  return
}

// -----

func @illegal_dimension() {
  // expected-error@+1 {{dimension "o" is invalid}}
  %gDimX = "gpu.grid_dim"() {dimension = "o"} : () -> (index)

  return
}

// -----

func @reduce_no_op_no_body(%arg0 : f32) {
  // expected-error@+1 {{expected either an op attribute or a non-empty body}}
  %res = "gpu.all_reduce"(%arg0) ({}) : (f32) -> (f32)
  return
}

// -----

func @reduce_op_and_body(%arg0 : f32) {
  // expected-error@+1 {{expected either an op attribute or a non-empty body}}
  %res = "gpu.all_reduce"(%arg0) ({
  ^bb(%lhs : f32, %rhs : f32):
    "gpu.yield"(%lhs) : (f32) -> ()
  }) {op = "add"} : (f32) -> (f32)
}

// -----

func @reduce_invalid_op(%arg0 : f32) {
  // expected-error@+1 {{gpu.all_reduce' op attribute 'op' failed to satisfy constraint}}
  %res = "gpu.all_reduce"(%arg0) ({}) {op = "foo"} : (f32) -> (f32)
  return
}

// -----

func @reduce_invalid_op_type(%arg0 : f32) {
  // expected-error@+1 {{`and` accumulator is only compatible with Integer type}}
  %res = "gpu.all_reduce"(%arg0) ({}) {op = "and"} : (f32) -> (f32)
  return
}

// -----

func @reduce_incorrect_region_arguments(%arg0 : f32) {
  // expected-error@+1 {{expected two region arguments}}
  %res = "gpu.all_reduce"(%arg0) ({
  ^bb(%lhs : f32):
    "gpu.yield"(%lhs) : (f32) -> ()
  }) : (f32) -> (f32)
}

// -----

func @reduce_incorrect_region_arguments(%arg0 : f32) {
  // expected-error@+1 {{incorrect region argument type}}
  %res = "gpu.all_reduce"(%arg0) ({
  ^bb(%lhs : f32, %rhs : i32):
    "gpu.yield"(%lhs) : (f32) -> ()
  }) : (f32) -> (f32)
}

// -----

func @reduce_incorrect_yield(%arg0 : f32) {
  // expected-error@+1 {{expected one gpu.yield operand}}
  %res = "gpu.all_reduce"(%arg0) ({
  ^bb(%lhs : f32, %rhs : f32):
    "gpu.yield"(%lhs, %rhs) : (f32, f32) -> ()
  }) : (f32) -> (f32)
}

// -----

func @reduce_incorrect_yield(%arg0 : f32) {
  // expected-error@+1 {{incorrect gpu.yield type}}
  %res = "gpu.all_reduce"(%arg0) ({
  ^bb(%lhs : f32, %rhs : f32):
    %one = constant 1 : i32
    "gpu.yield"(%one) : (i32) -> ()
  }) : (f32) -> (f32)
}

// -----

func @reduce_incorrect_yield(%arg0 : f32) {
  // expected-error@+1 {{expected gpu.yield op in region}}
  %res = "gpu.all_reduce"(%arg0) ({
  ^bb(%lhs : f32, %rhs : f32):
    return
  }) : (f32) -> (f32)
}

// -----

func @shuffle_mismatching_type(%arg0 : f32, %arg1 : i32, %arg2 : i32) {
  // expected-error@+1 {{'gpu.shuffle' op requires the same type for value operand and result}}
  %shfl, %pred = "gpu.shuffle"(%arg0, %arg1, %arg2) { mode = "xor" } : (f32, i32, i32) -> (i32, i1)
}

// -----

func @shuffle_unsupported_type(%arg0 : index, %arg1 : i32, %arg2 : i32) {
  // expected-error@+1 {{'gpu.shuffle' op requires value operand type to be f32 or i32}}
  %shfl, %pred = gpu.shuffle %arg0, %arg1, %arg2 xor : index
}

// -----

module {
  gpu.module @gpu_funcs {
    // expected-error @+1 {{custom op 'gpu.func' gpu.func requires named arguments}}
    gpu.func @kernel_1(f32, f32) {
    ^bb0(%arg0: f32):
      gpu.return
    }
  }
}

// -----

module {
  module @gpu_funcs attributes {gpu.kernel_module} {
    // expected-error @+1 {{requires 'type' attribute of function type}}
    "gpu.func"() ({
      gpu.return
    }) {sym_name="kernel_1", type=f32} : () -> ()
  }
}

// -----

module {
  module @gpu_funcs attributes {gpu.kernel_module} {
    // expected-error @+1 {{expected memref type in attribution}}
    gpu.func @kernel() workgroup(%0: i32) {
      gpu.return
    }
  }
}

// -----

module {
  module @gpu_funcs attributes {gpu.kernel_module} {
    // expected-error @+1 {{expected memory space 3 in attribution}}
    gpu.func @kernel() workgroup(%0: memref<4xf32>) {
      gpu.return
    }
  }
}

// -----

module {
  module @gpu_funcs attributes {gpu.kernel_module} {
    // expected-error @+1 {{expected memory space 5 in attribution}}
    gpu.func @kernel() private(%0: memref<4xf32>) {
      gpu.return
    }
  }
}

// -----

module {
  module @gpu_funcs attributes {gpu.kernel_module} {
    // expected-error @+1 {{expected memory space 5 in attribution}}
    gpu.func @kernel() private(%0: memref<4xf32>) {
      gpu.return
    }
  }
}

// -----

module {
  gpu.module @gpu_funcs {
    // expected-note @+1 {{return type declared here}}
    gpu.func @kernel() {
      %0 = constant 0 : index
      // expected-error @+1 {{'gpu.return' op expected 0 result operands}}
      gpu.return %0 : index
    }
  }
}

// -----

module {
  gpu.module @gpu_funcs {
    // expected-error @+1 {{'gpu.func' op expected void return type for kernel function}}
    gpu.func @kernel() -> index kernel {
      %0 = constant 0 : index
      gpu.return
    }
  }
}
