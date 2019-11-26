// RUN: mlir-opt -spirv-lower-abi-attrs -verify-diagnostics %s -o - | FileCheck %s

// CHECK-LABEL: spv.module
spv.module "Logical" "GLSL450" {
  // CHECK-DAG: spv.globalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize")
  spv.globalVariable @__builtin_var_WorkgroupSize__ built_in("WorkgroupSize") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.globalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
  spv.globalVariable @__builtin_var_NumWorkgroups__ built_in("NumWorkgroups") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.globalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId")
  spv.globalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.globalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  spv.globalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.globalVariable [[VAR0:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32 [4]> [16]> [0]>, StorageBuffer>
  // CHECK-DAG: spv.globalVariable [[VAR1:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32 [4]> [16]> [0]>, StorageBuffer>
  // CHECK-DAG: spv.globalVariable [[VAR2:@.*]] bind(0, 2) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32 [4]> [16]> [0]>, StorageBuffer>
  // CHECK-DAG: spv.globalVariable [[VAR3:@.*]] bind(0, 3) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK-DAG: spv.globalVariable [[VAR4:@.*]] bind(0, 4) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK-DAG: spv.globalVariable [[VAR5:@.*]] bind(0, 5) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK-DAG: spv.globalVariable [[VAR6:@.*]] bind(0, 6) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK: func [[FN:@.*]]()
  func @load_store_kernel(%arg0: !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32>>>, StorageBuffer>
                          {spirv.interface_var_abi = {binding = 0 : i32,
                                                      descriptor_set = 0 : i32,
                                                      storage_class = 12 : i32}},
                          %arg1: !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32>>>, StorageBuffer>
                          {spirv.interface_var_abi = {binding = 1 : i32,
                                                      descriptor_set = 0 : i32,
                                                      storage_class = 12 : i32}},
                          %arg2: !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32>>>, StorageBuffer>
                          {spirv.interface_var_abi = {binding = 2 : i32,
                                                      descriptor_set = 0 : i32,
                                                      storage_class = 12 : i32}},
                          %arg3: i32
                          {spirv.interface_var_abi = {binding = 3 : i32,
                                                      descriptor_set = 0 : i32,
                                                      storage_class = 12 : i32}},
                          %arg4: i32
                          {spirv.interface_var_abi = {binding = 4 : i32,
                                                      descriptor_set = 0 : i32,
                                                      storage_class = 12 : i32}},
                          %arg5: i32
                          {spirv.interface_var_abi = {binding = 5 : i32,
                                                      descriptor_set = 0 : i32,
                                                      storage_class = 12 : i32}},
                          %arg6: i32
                          {spirv.interface_var_abi = {binding = 6 : i32,
                                                      descriptor_set = 0 : i32,
                                                      storage_class = 12 : i32}})
  attributes  {spirv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    // CHECK: [[ADDRESSARG6:%.*]] = spv._address_of [[VAR6]]
    // CHECK: [[CONST6:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG6PTR:%.*]] = spv.AccessChain [[ADDRESSARG6]]{{\[}}[[CONST6]]
    // CHECK: {{%.*}} = spv.Load "StorageBuffer" [[ARG6PTR]]
    // CHECK: [[ADDRESSARG5:%.*]] = spv._address_of [[VAR5]]
    // CHECK: [[CONST5:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG5PTR:%.*]] = spv.AccessChain [[ADDRESSARG5]]{{\[}}[[CONST5]]
    // CHECK: {{%.*}} = spv.Load "StorageBuffer" [[ARG5PTR]]
    // CHECK: [[ADDRESSARG4:%.*]] = spv._address_of [[VAR4]]
    // CHECK: [[CONST4:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG4PTR:%.*]] = spv.AccessChain [[ADDRESSARG4]]{{\[}}[[CONST4]]
    // CHECK: [[ARG4:%.*]] = spv.Load "StorageBuffer" [[ARG4PTR]]
    // CHECK: [[ADDRESSARG3:%.*]] = spv._address_of [[VAR3]]
    // CHECK: [[CONST3:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG3PTR:%.*]] = spv.AccessChain [[ADDRESSARG3]]{{\[}}[[CONST3]]
    // CHECK: [[ARG3:%.*]] = spv.Load "StorageBuffer" [[ARG3PTR]]
    // CHECK: [[ARG2:%.*]] = spv._address_of [[VAR2]]
    // CHECK: [[ARG1:%.*]] = spv._address_of [[VAR1]]
    // CHECK: [[ARG0:%.*]] = spv._address_of [[VAR0]]
    %0 = spv._address_of @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %1 = spv.Load "Input" %0 : vector<3xi32>
    %2 = spv.CompositeExtract %1[0 : i32] : vector<3xi32>
    %3 = spv._address_of @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %4 = spv.Load "Input" %3 : vector<3xi32>
    %5 = spv.CompositeExtract %4[1 : i32] : vector<3xi32>
    %6 = spv._address_of @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %7 = spv.Load "Input" %6 : vector<3xi32>
    %8 = spv.CompositeExtract %7[2 : i32] : vector<3xi32>
    %9 = spv._address_of @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %10 = spv.Load "Input" %9 : vector<3xi32>
    %11 = spv.CompositeExtract %10[0 : i32] : vector<3xi32>
    %12 = spv._address_of @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %13 = spv.Load "Input" %12 : vector<3xi32>
    %14 = spv.CompositeExtract %13[1 : i32] : vector<3xi32>
    %15 = spv._address_of @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %16 = spv.Load "Input" %15 : vector<3xi32>
    %17 = spv.CompositeExtract %16[2 : i32] : vector<3xi32>
    %18 = spv._address_of @__builtin_var_NumWorkgroups__ : !spv.ptr<vector<3xi32>, Input>
    %19 = spv.Load "Input" %18 : vector<3xi32>
    %20 = spv.CompositeExtract %19[0 : i32] : vector<3xi32>
    %21 = spv._address_of @__builtin_var_NumWorkgroups__ : !spv.ptr<vector<3xi32>, Input>
    %22 = spv.Load "Input" %21 : vector<3xi32>
    %23 = spv.CompositeExtract %22[1 : i32] : vector<3xi32>
    %24 = spv._address_of @__builtin_var_NumWorkgroups__ : !spv.ptr<vector<3xi32>, Input>
    %25 = spv.Load "Input" %24 : vector<3xi32>
    %26 = spv.CompositeExtract %25[2 : i32] : vector<3xi32>
    %27 = spv._address_of @__builtin_var_WorkgroupSize__ : !spv.ptr<vector<3xi32>, Input>
    %28 = spv.Load "Input" %27 : vector<3xi32>
    %29 = spv.CompositeExtract %28[0 : i32] : vector<3xi32>
    %30 = spv._address_of @__builtin_var_WorkgroupSize__ : !spv.ptr<vector<3xi32>, Input>
    %31 = spv.Load "Input" %30 : vector<3xi32>
    %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
    %33 = spv._address_of @__builtin_var_WorkgroupSize__ : !spv.ptr<vector<3xi32>, Input>
    %34 = spv.Load "Input" %33 : vector<3xi32>
    %35 = spv.CompositeExtract %34[2 : i32] : vector<3xi32>
    // CHECK: spv.IAdd [[ARG3]]
    %36 = spv.IAdd %arg3, %2 : i32
    // CHECK: spv.IAdd [[ARG4]]
    %37 = spv.IAdd %arg4, %11 : i32
    // CHECK: spv.AccessChain [[ARG0]]
    %c0 = spv.constant 0 : i32
    %38 = spv.AccessChain %arg0[%c0, %36, %37] : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32>>>, StorageBuffer>
    %39 = spv.Load "StorageBuffer" %38 : f32
    // CHECK: spv.AccessChain [[ARG1]]
    %40 = spv.AccessChain %arg1[%c0, %36, %37] : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32>>>, StorageBuffer>
    %41 = spv.Load "StorageBuffer" %40 : f32
    %42 = spv.FAdd %39, %41 : f32
    // CHECK: spv.AccessChain [[ARG2]]
    %43 = spv.AccessChain %arg2[%c0, %36, %37] : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<4 x f32>>>, StorageBuffer>
    spv.Store "StorageBuffer" %43, %42 : f32
    spv.Return
  }
  // CHECK: spv.EntryPoint "GLCompute" [[FN]], [[WORKGROUPID]], [[LOCALINVOCATIONID]], [[NUMWORKGROUPS]], [[WORKGROUPSIZE]]
  // CHECK-NEXT: spv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
} attributes {capabilities = ["Shader"], extensions = ["SPV_KHR_storage_buffer_storage_class"]}
