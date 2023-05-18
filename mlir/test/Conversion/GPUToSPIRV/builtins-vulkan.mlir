// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv="use-64bit-index=false" %s -o - | FileCheck %s --check-prefix=INDEX32
// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv="use-64bit-index=true" %s -o - | FileCheck %s --check-prefix=INDEX64

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  // INDEX64-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX64: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      // INDEX64: spirv.UConvert %{{.+}} : i32 to i64
      %0 = gpu.block_id x
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    %c256 = arith.constant 256 : i32
    gpu.launch_func @kernels::@builtin_workgroup_id_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
        dynamic_shared_memory_size %c256
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
      %0 = gpu.block_id y
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
      %0 = gpu.block_id z
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
      // The constant value is obtained from the spirv.entry_point_abi.
      // Note that this ignores the workgroup size specification in gpu.launch.
      // We may want to define gpu.workgroup_size and convert it to the entry
      // point ABI we want here.
      // INDEX32: spirv.Constant 32 : i32
      %0 = gpu.block_dim x
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // The constant value is obtained from the spirv.entry_point_abi.
      // INDEX32: spirv.Constant 4 : i32
      %0 = gpu.block_dim y
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // The constant value is obtained from the spirv.entry_point_abi.
      // INDEX32: spirv.Constant 1 : i32
      %0 = gpu.block_dim z
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_local_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_local_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[LOCALINVOCATIONID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.thread_id x
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_num_workgroups_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_num_workgroups_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[NUMWORKGROUPS]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.grid_dim x
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[SUBGROUPID:@.*]] built_in("SubgroupId") : !spirv.ptr<i32, Input>
  gpu.module @kernels {
    gpu.func @builtin_subgroup_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[SUBGROUPID]]
      // INDEX32-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      %0 = gpu.subgroup_id : index
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[NUMSUBGROUPS:@.*]] built_in("NumSubgroups") : !spirv.ptr<i32, Input>
  gpu.module @kernels {
    gpu.func @builtin_num_subgroups() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[NUMSUBGROUPS]]
      // INDEX32-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      %0 = gpu.num_subgroups : index
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}}
  // INDEX32: spirv.GlobalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPSIZE]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.block_dim x
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}}
  // INDEX32: spirv.GlobalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPSIZE]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
      %0 = gpu.block_dim y
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}}
  // INDEX32: spirv.GlobalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[WORKGROUPSIZE]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
      %0 = gpu.block_dim z
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_global_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[GLOBALINVOCATIONID:@.*]] built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_global_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[GLOBALINVOCATIONID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.global_id x
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_global_id_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[GLOBALINVOCATIONID:@.*]] built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_global_id_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[GLOBALINVOCATIONID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
      %0 = gpu.global_id y
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_global_id_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[GLOBALINVOCATIONID:@.*]] built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  gpu.module @kernels {
    gpu.func @builtin_global_id_z() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[GLOBALINVOCATIONID]]
      // INDEX32-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]]
      // INDEX32-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
      %0 = gpu.global_id z
      gpu.return
    }
  }
}


// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, Int64], []>, #spirv.resource_limits<>>
} {
  // INDEX32-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX32: spirv.GlobalVariable [[SUBGROUPSIZE:@.*]] built_in("SubgroupSize") : !spirv.ptr<i32, Input>
  // INDEX64-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // INDEX64: spirv.GlobalVariable [[SUBGROUPSIZE:@.*]] built_in("SubgroupSize") : !spirv.ptr<i32, Input>
  gpu.module @kernels {
    gpu.func @builtin_subgroup_size() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // INDEX32: [[ADDRESS:%.*]] = spirv.mlir.addressof [[SUBGROUPSIZE]]
      // INDEX32-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]]
      // INDEX64: spirv.UConvert %{{.+}} : i32 to i64
      %0 = gpu.subgroup_size : index
      gpu.return
    }
  }
}
