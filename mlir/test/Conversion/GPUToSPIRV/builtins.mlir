// RUN: mlir-opt -split-input-file -pass-pipeline='convert-gpu-to-spirv{workgroup-size=32,4}' %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_workgroup_id_x", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_x()
      attributes {gpu.kernel} {
      // CHECK: [[ADDRESS:%.*]] = spv._address_of [[WORKGROUPID]]
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
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_workgroup_id_y", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_y()
      attributes {gpu.kernel} {
      // CHECK: [[ADDRESS:%.*]] = spv._address_of [[WORKGROUPID]]
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
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_workgroup_id_z", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_z()
      attributes {gpu.kernel} {
      // CHECK: [[ADDRESS:%.*]] = spv._address_of [[WORKGROUPID]]
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
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_workgroup_size_x", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x()
      attributes {gpu.kernel} {
      // The constant value is obtained fomr the command line option above.
      // CHECK: spv.constant 32 : i32
      %0 = "gpu.block_dim"() {dimension = "x"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_workgroup_size_y", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_y()
      attributes {gpu.kernel} {
      // The constant value is obtained fomr the command line option above.
      // CHECK: spv.constant 4 : i32
      %0 = "gpu.block_dim"() {dimension = "y"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_workgroup_size_z", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_z()
      attributes {gpu.kernel} {
      // The constant value is obtained fomr the command line option above (1 is default).
      // CHECK: spv.constant 1 : i32
      %0 = "gpu.block_dim"() {dimension = "z"} : () -> index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_local_id_x", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_local_id_x()
      attributes {gpu.kernel} {
      // CHECK: [[ADDRESS:%.*]] = spv._address_of [[LOCALINVOCATIONID]]
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
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_num_workgroups_x", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
  gpu.module @kernels {
    gpu.func @builtin_num_workgroups_x()
      attributes {gpu.kernel} {
      // CHECK: [[ADDRESS:%.*]] = spv._address_of [[NUMWORKGROUPS]]
      // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = "gpu.grid_dim"() {dimension = "x"} : () -> index
      gpu.return
    }
  }
}
