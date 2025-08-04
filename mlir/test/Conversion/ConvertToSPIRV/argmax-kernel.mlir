// RUN: mlir-opt -test-convert-to-spirv -cse %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, Groups, GroupNonUniformArithmetic, GroupNonUniformBallot], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: spirv.module @{{.*}} Logical GLSL450
  // CHECK-DAG:   spirv.GlobalVariable @[[$LOCALINVOCATIONIDVAR:.*]] built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  // CHECK-DAG:   spirv.GlobalVariable @[[$SUBGROUPSIZE:.*]] built_in("SubgroupSize") : !spirv.ptr<i32, Input>
  // CHECK-LABEL: spirv.func @argmax
  // CHECK-SAME:  %[[ARG0:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
  // CHECK-SAME:  %[[ARG1:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
  // CHECK-SAME:  %[[ARG2:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}
  gpu.module @kernels {
    gpu.func @argmax(%input : memref<128xf32>, %output : memref<1xi32>, %total_count_buf : memref<1xi32>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
      // CHECK: %[[C0:.*]] = spirv.Constant 0 : i32
      // CHECK: %[[C1:.*]] = spirv.Constant 1 : i32
      // CHECK: %[[ADDRESSSUBGROUPSIZE:.*]] = spirv.mlir.addressof @[[$SUBGROUPSIZE]]
      // CHECK: %[[SUBGROUPSIZE:.*]] = spirv.Load "Input" %[[ADDRESSSUBGROUPSIZE]]
      // CHECK: %[[ADDRESSLOCALINVOCATIONID:.*]] = spirv.mlir.addressof @[[$LOCALINVOCATIONIDVAR]]
      // CHECK: %[[LOCALINVOCATIONID:.*]] = spirv.Load "Input" %[[ADDRESSLOCALINVOCATIONID]]
      // CHECK: %[[LOCALINVOCATIONIDX:.*]] = spirv.CompositeExtract %[[LOCALINVOCATIONID]]{{\[}}0 : i32{{\]}}
      // CHECK: %[[AC:.*]] = spirv.AccessChain %[[ARG2]][%[[C0]], %[[C0]]] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
      // CHECK: %[[LOAD:.*]] = spirv.Load "StorageBuffer" %[[AC]] : i32
      // CHECK: %[[AC0:.*]] = spirv.AccessChain %[[ARG0]][%[[C0]], %[[LOCALINVOCATIONIDX]]] : !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
      // CHECK: %[[LOAD0:.*]] = spirv.Load "StorageBuffer" %[[AC0]] : f32
      // CHECK: %[[UB:.*]] = spirv.UDiv %[[LOAD]], %[[SUBGROUPSIZE]] : i32
      // CHECK: %[[FUNC0:.*]] = spirv.Variable : !spirv.ptr<i32, Function>
      // CHECK: %[[FUNC1:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
      %idx_0 = arith.constant 0 : index
      %idx_1 = arith.constant 1 : index
      %lane_count_idx = gpu.subgroup_size : index
      %lane_count_i32 = index.castu %lane_count_idx : index to i32
      %lane_id_idx = gpu.thread_id x
      %lane_id_i32 = index.castu %lane_id_idx : index to i32
      %total_count = memref.load %total_count_buf[%idx_0] : memref<1xi32>
      %lane_res_init = arith.constant 0 : i32
      %lane_max_init = memref.load %input[%lane_id_idx] : memref<128xf32>
      %num_batches_i32 = arith.divui %total_count, %lane_count_i32 : i32
      %num_batches_idx = index.castu %num_batches_i32 : i32 to index

      // CHECK: spirv.mlir.loop {
      // CHECK:   spirv.Branch ^[[HEADER:.*]](%[[C1]], %[[C0]], %[[LOAD0]] : i32, i32, f32)
      // CHECK: ^[[HEADER]](%[[INDVAR0:.*]]: i32, %[[INDVAR1:.*]]: i32, %[[INDVAR2:.*]]: f32):
      // CHECK:   %[[SLT:.*]] = spirv.SLessThan %[[INDVAR0]], %[[UB]] : i32
      // CHECK:   spirv.BranchConditional %[[SLT]], ^[[BODY:.*]], ^[[MERGE:.*]]
      // CHECK: ^[[BODY]]:
      // CHECK:   %[[MUL:.*]] = spirv.IMul %[[SUBGROUPSIZE]], %[[INDVAR0]] : i32
      // CHECK:   %[[ADD:.*]] = spirv.IAdd %[[MUL]], %[[LOCALINVOCATIONIDX]] : i32
      // CHECK:   %[[AC1:.*]] = spirv.AccessChain %[[ARG0]][%[[C0]], %[[ADD]]] : !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
      // CHECK:   %[[LOAD1:.*]] = spirv.Load "StorageBuffer" %[[AC1]] : f32
      // CHECK:   %[[OGT:.*]] = spirv.FOrdGreaterThan %[[LOAD1]], %[[INDVAR2]] : f32
      // CHECK:   %[[SELECT0:.*]] = spirv.Select %[[OGT]], %[[ADD]], %[[INDVAR1]] : i1, i32
      // CHECK:   %[[SELECT1:.*]] = spirv.Select %[[OGT]], %[[LOAD1]], %[[INDVAR2]] : i1, f32
      // CHECK:   spirv.Store "Function" %[[FUNC0]], %[[SELECT0]] : i32
      // CHECK:   spirv.Store "Function" %[[FUNC1]], %[[SELECT1]] : f32
      // CHECK:   %[[ADD1:.*]] = spirv.IAdd %[[INDVAR0]], %[[C1]] : i32
      // CHECK:   spirv.Branch ^[[HEADER]](%[[ADD1]], %[[SELECT0]], %[[SELECT1]] : i32, i32, f32)
      // CHECK: ^[[MERGE]]:
      // CHECK:   spirv.mlir.merge
      // CHECK: }
      // CHECK-DAG: %[[LANE_RES:.*]] = spirv.Load "Function" %[[FUNC0]] : i32
      // CHECK-DAG: %[[LANE_MAX:.*]] = spirv.Load "Function" %[[FUNC1]] : f32
      %lane_res, %lane_max = scf.for %iter = %idx_1 to %num_batches_idx step %idx_1
      iter_args(%lane_res_iter = %lane_res_init, %lane_max_iter = %lane_max_init) -> (i32, f32) {
        %iter_i32 = index.castu %iter : index to i32
        %mul = arith.muli %lane_count_i32, %iter_i32 : i32
        %idx_i32 = arith.addi %mul, %lane_id_i32 : i32
        %idx = index.castu %idx_i32 : i32 to index
        %elem = memref.load %input[%idx] : memref<128xf32>
        %gt = arith.cmpf ogt, %elem, %lane_max_iter : f32
        %lane_res_next = arith.select %gt, %idx_i32, %lane_res_iter : i32
        %lane_max_next = arith.select %gt, %elem, %lane_max_iter : f32
        scf.yield %lane_res_next, %lane_max_next : i32, f32
      }

      // CHECK: %[[SUBGROUP_MAX:.*]] = spirv.GroupNonUniformFMax <Subgroup> <Reduce> %[[LANE_MAX]] : f32 -> f32
      // CHECK: %[[OEQ:.*]] = spirv.FOrdEqual %[[LANE_MAX]], %[[SUBGROUP_MAX]] : f32
      // CHECK: %[[BALLOT:.*]] = spirv.GroupNonUniformBallot <Subgroup> %[[OEQ]] : vector<4xi32>
      // CHECK: %[[BALLOTLSB:.*]] = spirv.GroupNonUniformBallotFindLSB <Subgroup> %[[BALLOT]] : vector<4xi32>, i32
      // CHECK: %[[EQ:.*]] = spirv.IEqual %[[BALLOTLSB]], %[[LOCALINVOCATIONIDX]] : i32
      %subgroup_max = gpu.subgroup_reduce maximumf %lane_max : (f32) -> (f32)
      %eq = arith.cmpf oeq, %lane_max, %subgroup_max : f32
      %ballot = spirv.GroupNonUniformBallot <Subgroup> %eq : vector<4xi32>
      %lsb = spirv.GroupNonUniformBallotFindLSB <Subgroup> %ballot : vector<4xi32>, i32
      %cond = arith.cmpi eq, %lsb, %lane_id_i32 : i32

      // CHECK: spirv.mlir.selection {
      // CHECK:   spirv.BranchConditional %[[EQ]], ^[[TRUE:.*]], ^[[FALSE:.*]]
      // CHECK: ^[[TRUE]]:
      // CHECK:   %[[AC2:.*]] = spirv.AccessChain %[[ARG1]][%[[C0]], %[[C0]]] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
      // CHECK:   spirv.Store "StorageBuffer" %[[AC2]], %[[LANE_RES]] : i32
      // CHECK:   spirv.Branch ^[[FALSE]]
      // CHECK: ^[[FALSE]]:
      // CHECK:   spirv.mlir.merge
      // CHECK: }
      scf.if %cond {
        memref.store %lane_res, %output[%idx_0] : memref<1xi32>
      }

      // CHECK: spirv.Return
      gpu.return
    }
  }
}
