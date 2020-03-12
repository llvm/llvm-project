// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  func @builtin() {
    %c0 = constant 1 : index
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0) {kernel = "builtin_workgroup_id_x", kernel_module = @kernels} : (index, index, index, index, index, index) -> ()
    return
  }

  // CHECK-LABEL:  spv.module Logical GLSL450
  // CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_x()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
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

  // CHECK-LABEL:  spv.module Logical GLSL450
  // CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_y()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
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

  // CHECK-LABEL:  spv.module Logical GLSL450
  // CHECK: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_id_z()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
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

  // CHECK-LABEL:  spv.module Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[32, 1, 1]>: vector<3xi32>}} {
      // The constant value is obtained from the spv.entry_point_abi.
      // Note that this ignores the workgroup size specification in gpu.launch.
      // We may want to define gpu.workgroup_size and convert it to the entry
      // point ABI we want here.
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

  // CHECK-LABEL:  spv.module Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_y()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // The constant value is obtained from the spv.entry_point_abi.
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

  // CHECK-LABEL:  spv.module Logical GLSL450
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_z()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // The constant value is obtained from the spv.entry_point_abi.
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

  // CHECK-LABEL:  spv.module Logical GLSL450
  // CHECK: spv.globalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_local_id_x()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
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

  // CHECK-LABEL:  spv.module Logical GLSL450
  // CHECK: spv.globalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
  gpu.module @kernels {
    gpu.func @builtin_num_workgroups_x()
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>}} {
      // CHECK: [[ADDRESS:%.*]] = spv._address_of [[NUMWORKGROUPS]]
      // CHECK-NEXT: [[VEC:%.*]] = spv.Load "Input" [[ADDRESS]]
      // CHECK-NEXT: {{%.*}} = spv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = "gpu.grid_dim"() {dimension = "x"} : () -> index
      gpu.return
    }
  }
}
