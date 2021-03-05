// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_x() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[WORKGROUPID]]
      // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_y() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[WORKGROUPID]]
      // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}1 : i32{{\]}}
      %0 = "gpu.block_id"() {dimension = "y"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_id_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_z() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[WORKGROUPID]]
      // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}2 : i32{{\]}}
      %0 = "gpu.block_id"() {dimension = "z"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]>: vector<3xi32>}} {
      // The constant value is obtained from the spv.entry_point_abi.
      // Note that this ignores the workgroup size specification in gpu.launch.
      // We may want to define gpu.workgroup_size and convert it to the entry
      // point ABI we want here.
      // CHECK: spv.Constant 32 : i32
      %0 = "gpu.block_dim"() {dimension = "x"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_y
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_y() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // The constant value is obtained from the spv.entry_point_abi.
      // CHECK: spv.Constant 4 : i32
      %0 = "gpu.block_dim"() {dimension = "y"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_z
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_z() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // The constant value is obtained from the spv.entry_point_abi.
      // CHECK: spv.Constant 1 : i32
      %0 = "gpu.block_dim"() {dimension = "z"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_local_id_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_local_id_x() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[LOCALINVOCATIONID]]
      // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    gpu.launch_func @kernels::@builtin_num_workgroups_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
  gpu.module @kernels {
    gpu.func @builtin_num_workgroups_x() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[NUMWORKGROUPS]]
      // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = "gpu.grid_dim"() {dimension = "x"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[SUBGROUPID:@.*]] built_in("SubgroupId")
  gpu.module @kernels {
    gpu.func @builtin_subgroup_id() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[SUBGROUPID]]
      // CHECK-NEXT: {{%.*}} = spv.Load "Input" [[ADDRESS]]
      %0 = gpu.subgroup_id : index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[NUMSUBGROUPS:@.*]] built_in("NumSubgroups")
  gpu.module @kernels {
    gpu.func @builtin_num_subgroups() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[NUMSUBGROUPS]]
      // CHECK-NEXT: {{%.*}} = spv.Load "Input" [[ADDRESS]]
      %0 = gpu.num_subgroups : index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK-LABEL:  spv.module @{{.*}} Logical GLSL450
  // CHECK: spv.GlobalVariable [[SUBGROUPSIZE:@.*]] built_in("SubgroupSize")
  gpu.module @kernels {
    gpu.func @builtin_subgroup_size() kernel
      attributes {spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv.mlir.addressof [[SUBGROUPSIZE]]
      // CHECK-NEXT: {{%.*}} = spv.Load "Input" [[ADDRESS]]
      %0 = gpu.subgroup_size : index
      gpu.return
    }
  }
}
