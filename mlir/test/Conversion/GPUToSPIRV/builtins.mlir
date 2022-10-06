// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPID]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.block_id x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    %c256 = arith.constant 256 : i32
    gpu.launch_func @kernels::@builtin_workgroup_id_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
        dynamic_shared_memory_size %c256
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPID]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
      %0 = gpu.block_id y
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPID]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
      %0 = gpu.block_id z
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 1, 1]>: vector<3xi32>>} {
      // The constant value is obtained from the spirv.entry_point_abi.
      // Note that this ignores the workgroup size specification in gpu.launch.
      // We may want to define gpu.workgroup_size and convert it to the entry
      // point ABI we want here.
      // CHECK: spirv.Constant 32 : i32
      %0 = gpu.block_dim x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 4, 1]>: vector<3xi32>>} {
      // The constant value is obtained from the spirv.entry_point_abi.
      // CHECK: spirv.Constant 4 : i32
      %0 = gpu.block_dim y
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 4, 1]>: vector<3xi32>>} {
      // The constant value is obtained from the spirv.entry_point_abi.
      // CHECK: spirv.Constant 1 : i32
      %0 = gpu.block_dim z
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_local_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_local_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[LOCALINVOCATIONID]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.thread_id x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_num_workgroups_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
  gpu.module @kernels {
    gpu.func @builtin_num_workgroups_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[NUMWORKGROUPS]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.grid_dim x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[SUBGROUPID:@.*]] built_in("SubgroupId")
  gpu.module @kernels {
    gpu.func @builtin_subgroup_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[SUBGROUPID]]
      // CHECK-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      %0 = gpu.subgroup_id : index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[NUMSUBGROUPS:@.*]] built_in("NumSubgroups")
  gpu.module @kernels {
    gpu.func @builtin_num_subgroups() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[NUMSUBGROUPS]]
      // CHECK-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      %0 = gpu.num_subgroups : index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPSIZE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.block_dim x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPSIZE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
      %0 = gpu.block_dim y
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPSIZE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
      %0 = gpu.block_dim z
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_global_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[GLOBALINVOCATIONID:@.*]] built_in("GlobalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_global_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[GLOBALINVOCATIONID]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.global_id x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_global_id_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[GLOBALINVOCATIONID:@.*]] built_in("GlobalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_global_id_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[GLOBALINVOCATIONID]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
      %0 = gpu.global_id y
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_global_id_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[GLOBALINVOCATIONID:@.*]] built_in("GlobalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_global_id_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[GLOBALINVOCATIONID]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
      %0 = gpu.global_id z
      gpu.return
    }
  }
}


// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK: spirv.GlobalVariable [[SUBGROUPSIZE:@.*]] built_in("SubgroupSize")
  gpu.module @kernels {
    gpu.func @builtin_subgroup_size() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[SUBGROUPSIZE]]
      // CHECK-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      %0 = gpu.subgroup_size : index
      gpu.return
    }
  }
}
