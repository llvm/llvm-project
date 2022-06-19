// RUN: mlir-opt -spirv-lower-abi-attrs -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>
} {

// CHECK-LABEL: spv.module
spv.module Logical GLSL450 {
  // CHECK-DAG: spv.GlobalVariable [[WORKGROUPSIZE:@.*]] built_in("WorkgroupSize")
  spv.GlobalVariable @__builtin_var_WorkgroupSize__ built_in("WorkgroupSize") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.GlobalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
  spv.GlobalVariable @__builtin_var_NumWorkgroups__ built_in("NumWorkgroups") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.GlobalVariable [[LOCALINVOCATIONID:@.*]] built_in("LocalInvocationId")
  spv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.GlobalVariable [[WORKGROUPID:@.*]] built_in("WorkgroupId")
  spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.GlobalVariable [[VAR0:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32, stride=4>, stride=16> [0])>, StorageBuffer>
  // CHECK-DAG: spv.GlobalVariable [[VAR1:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32, stride=4>, stride=16> [0])>, StorageBuffer>
  // CHECK-DAG: spv.GlobalVariable [[VAR2:@.*]] bind(0, 2) : !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32, stride=4>, stride=16> [0])>, StorageBuffer>
  // CHECK-DAG: spv.GlobalVariable [[VAR3:@.*]] bind(0, 3) : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>
  // CHECK-DAG: spv.GlobalVariable [[VAR4:@.*]] bind(0, 4) : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>
  // CHECK-DAG: spv.GlobalVariable [[VAR5:@.*]] bind(0, 5) : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>
  // CHECK-DAG: spv.GlobalVariable [[VAR6:@.*]] bind(0, 6) : !spv.ptr<!spv.struct<(i32 [0])>, StorageBuffer>
  // CHECK: spv.func [[FN:@.*]]()
  spv.func @load_store_kernel(
    %arg0: !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32>>)>, StorageBuffer>
    {spv.interface_var_abi = #spv.interface_var_abi<(0, 0)>},
    %arg1: !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32>>)>, StorageBuffer>
    {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>},
    %arg2: !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32>>)>, StorageBuffer>
    {spv.interface_var_abi = #spv.interface_var_abi<(0, 2)>},
    %arg3: i32
    {spv.interface_var_abi = #spv.interface_var_abi<(0, 3), StorageBuffer>},
    %arg4: i32
    {spv.interface_var_abi = #spv.interface_var_abi<(0, 4), StorageBuffer>},
    %arg5: i32
    {spv.interface_var_abi = #spv.interface_var_abi<(0, 5), StorageBuffer>},
    %arg6: i32
    {spv.interface_var_abi = #spv.interface_var_abi<(0, 6), StorageBuffer>}) "None"
  attributes  {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[32, 1, 1]> : vector<3xi32>>} {
    // CHECK: [[ADDRESSARG6:%.*]] = spv.mlir.addressof [[VAR6]]
    // CHECK: [[CONST6:%.*]] = spv.Constant 0 : i32
    // CHECK: [[ARG6PTR:%.*]] = spv.AccessChain [[ADDRESSARG6]]{{\[}}[[CONST6]]
    // CHECK: {{%.*}} = spv.Load "StorageBuffer" [[ARG6PTR]]
    // CHECK: [[ADDRESSARG5:%.*]] = spv.mlir.addressof [[VAR5]]
    // CHECK: [[CONST5:%.*]] = spv.Constant 0 : i32
    // CHECK: [[ARG5PTR:%.*]] = spv.AccessChain [[ADDRESSARG5]]{{\[}}[[CONST5]]
    // CHECK: {{%.*}} = spv.Load "StorageBuffer" [[ARG5PTR]]
    // CHECK: [[ADDRESSARG4:%.*]] = spv.mlir.addressof [[VAR4]]
    // CHECK: [[CONST4:%.*]] = spv.Constant 0 : i32
    // CHECK: [[ARG4PTR:%.*]] = spv.AccessChain [[ADDRESSARG4]]{{\[}}[[CONST4]]
    // CHECK: [[ARG4:%.*]] = spv.Load "StorageBuffer" [[ARG4PTR]]
    // CHECK: [[ADDRESSARG3:%.*]] = spv.mlir.addressof [[VAR3]]
    // CHECK: [[CONST3:%.*]] = spv.Constant 0 : i32
    // CHECK: [[ARG3PTR:%.*]] = spv.AccessChain [[ADDRESSARG3]]{{\[}}[[CONST3]]
    // CHECK: [[ARG3:%.*]] = spv.Load "StorageBuffer" [[ARG3PTR]]
    // CHECK: [[ADDRESSARG2:%.*]] = spv.mlir.addressof [[VAR2]]
    // CHECK: [[ARG2:%.*]] = spv.Bitcast [[ADDRESSARG2]]
    // CHECK: [[ADDRESSARG1:%.*]] = spv.mlir.addressof [[VAR1]]
    // CHECK: [[ARG1:%.*]] = spv.Bitcast [[ADDRESSARG1]]
    // CHECK: [[ADDRESSARG0:%.*]] = spv.mlir.addressof [[VAR0]]
    // CHECK: [[ARG0:%.*]] = spv.Bitcast [[ADDRESSARG0]]
    %0 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %1 = spv.Load "Input" %0 : vector<3xi32>
    %2 = spv.CompositeExtract %1[0 : i32] : vector<3xi32>
    %3 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %4 = spv.Load "Input" %3 : vector<3xi32>
    %5 = spv.CompositeExtract %4[1 : i32] : vector<3xi32>
    %6 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi32>, Input>
    %7 = spv.Load "Input" %6 : vector<3xi32>
    %8 = spv.CompositeExtract %7[2 : i32] : vector<3xi32>
    %9 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %10 = spv.Load "Input" %9 : vector<3xi32>
    %11 = spv.CompositeExtract %10[0 : i32] : vector<3xi32>
    %12 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %13 = spv.Load "Input" %12 : vector<3xi32>
    %14 = spv.CompositeExtract %13[1 : i32] : vector<3xi32>
    %15 = spv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spv.ptr<vector<3xi32>, Input>
    %16 = spv.Load "Input" %15 : vector<3xi32>
    %17 = spv.CompositeExtract %16[2 : i32] : vector<3xi32>
    %18 = spv.mlir.addressof @__builtin_var_NumWorkgroups__ : !spv.ptr<vector<3xi32>, Input>
    %19 = spv.Load "Input" %18 : vector<3xi32>
    %20 = spv.CompositeExtract %19[0 : i32] : vector<3xi32>
    %21 = spv.mlir.addressof @__builtin_var_NumWorkgroups__ : !spv.ptr<vector<3xi32>, Input>
    %22 = spv.Load "Input" %21 : vector<3xi32>
    %23 = spv.CompositeExtract %22[1 : i32] : vector<3xi32>
    %24 = spv.mlir.addressof @__builtin_var_NumWorkgroups__ : !spv.ptr<vector<3xi32>, Input>
    %25 = spv.Load "Input" %24 : vector<3xi32>
    %26 = spv.CompositeExtract %25[2 : i32] : vector<3xi32>
    %27 = spv.mlir.addressof @__builtin_var_WorkgroupSize__ : !spv.ptr<vector<3xi32>, Input>
    %28 = spv.Load "Input" %27 : vector<3xi32>
    %29 = spv.CompositeExtract %28[0 : i32] : vector<3xi32>
    %30 = spv.mlir.addressof @__builtin_var_WorkgroupSize__ : !spv.ptr<vector<3xi32>, Input>
    %31 = spv.Load "Input" %30 : vector<3xi32>
    %32 = spv.CompositeExtract %31[1 : i32] : vector<3xi32>
    %33 = spv.mlir.addressof @__builtin_var_WorkgroupSize__ : !spv.ptr<vector<3xi32>, Input>
    %34 = spv.Load "Input" %33 : vector<3xi32>
    %35 = spv.CompositeExtract %34[2 : i32] : vector<3xi32>
    // CHECK: spv.IAdd [[ARG3]]
    %36 = spv.IAdd %arg3, %2 : i32
    // CHECK: spv.IAdd [[ARG4]]
    %37 = spv.IAdd %arg4, %11 : i32
    // CHECK: spv.AccessChain [[ARG0]]
    %c0 = spv.Constant 0 : i32
    %38 = spv.AccessChain %arg0[%c0, %36, %37] : !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32>>)>, StorageBuffer>, i32, i32, i32
    %39 = spv.Load "StorageBuffer" %38 : f32
    // CHECK: spv.AccessChain [[ARG1]]
    %40 = spv.AccessChain %arg1[%c0, %36, %37] : !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32>>)>, StorageBuffer>, i32, i32, i32
    %41 = spv.Load "StorageBuffer" %40 : f32
    %42 = spv.FAdd %39, %41 : f32
    // CHECK: spv.AccessChain [[ARG2]]
    %43 = spv.AccessChain %arg2[%c0, %36, %37] : !spv.ptr<!spv.struct<(!spv.array<12 x !spv.array<4 x f32>>)>, StorageBuffer>, i32, i32, i32
    spv.Store "StorageBuffer" %43, %42 : f32
    spv.Return
  }
  // CHECK: spv.EntryPoint "GLCompute" [[FN]], [[WORKGROUPID]], [[LOCALINVOCATIONID]], [[NUMWORKGROUPS]], [[WORKGROUPSIZE]]
  // CHECK-NEXT: spv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
} // end spv.module

} // end module
