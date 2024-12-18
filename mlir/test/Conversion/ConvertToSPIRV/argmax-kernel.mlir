// RUN: mlir-opt -convert-to-spirv -cse %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, Groups, GroupNonUniformArithmetic, GroupNonUniformBallot], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: spirv.module @{{.*}} Logical GLSL450
  // CHECK-DAG: spirv.GlobalVariable @[[$LOCALINVOCATIONIDVAR:.*]] built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  // CHECK-LABEL: spirv.func @argmax
  // CHECK-SAME: %[[ARG0:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK-SAME: %[[ARG1:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>
  gpu.module @kernels {
    gpu.func @argmax(%input : memref<4xf32>, %output : memref<i32>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
      // CHECK: %[[C0:.*]] = spirv.Constant 0 : i32
      // CHECK: %[[C1:.*]] = spirv.Constant 1 : i32
      // CHECK: %[[C32:.*]] = spirv.Constant 32 : i32
      // CHECK: %[[ADDRESSLOCALINVOCATIONID:.*]] = spirv.mlir.addressof @[[$LOCALINVOCATIONIDVAR]]
      // CHECK: %[[LOCALINVOCATIONID:.*]] = spirv.Load "Input" %[[ADDRESSLOCALINVOCATIONID]]
      // CHECK: %[[LOCALINVOCATIONIDX:.*]] = spirv.CompositeExtract %[[LOCALINVOCATIONID]]{{\[}}0 : i32{{\]}}
      // CHECK: %[[AC0:.*]] = spirv.AccessChain %[[ARG0]][%[[C0]], %[[LOCALINVOCATIONIDX]]] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
      // CHECK: %[[LOAD0:.*]] = spirv.Load "StorageBuffer" %[[AC0]] : f32
      // CHECK: %[[FUNC0:.*]] = spirv.Variable : !spirv.ptr<i32, Function>
      // CHECK: %[[FUNC1:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
      %cst_0_idx = arith.constant 0 : index
      %cst_1_i32 = arith.constant 1 : i32
      %cst_1_idx = arith.constant 1 : index
      %cst_32 = arith.constant 32 : i32
      %num_batches = arith.divui %cst_1_i32, %cst_32 : i32
      %tx = gpu.thread_id x
      %tx_i32 = index.castu %tx : index to i32
      %ub = index.castu %num_batches : i32 to index
      %lane_res_init = arith.constant 0 : i32
      %lane_max_init = memref.load %input[%tx] : memref<4xf32>

      // CHECK: spirv.mlir.loop {
      // CHECK:   spirv.Branch ^[[HEADER:.*]](%[[C1]], %[[C0]], %[[LOAD0]] : i32, i32, f32)
      // CHECK: ^[[HEADER]](%[[INDVAR0:.*]]: i32, %[[INDVAR1:.*]]: i32, %[[INDVAR2:.*]]: f32):
      // CHECK:   %[[SLT:.*]] = spirv.SLessThan %[[INDVAR0]], %[[C0]] : i32
      // CHECK:   spirv.BranchConditional %[[SLT]], ^[[BODY:.*]], ^[[MERGE:.*]]
      // CHECK: ^[[BODY]]:
      // CHECK:   %[[MUL:.*]] = spirv.IMul %[[INDVAR0]], %[[C32]] : i32
      // CHECK:   %[[ADD:.*]] = spirv.IAdd %[[MUL]], %[[LOCALINVOCATIONIDX]] : i32
      // CHECK:   %[[AC1:.*]] = spirv.AccessChain %[[ARG0]][%[[C0]], %[[ADD]]] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
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
      %lane_res, %lane_max = scf.for %iter = %cst_1_idx to %ub step %cst_1_idx
      iter_args(%lane_res_iter = %lane_res_init, %lane_max_iter = %lane_max_init) -> (i32, f32) {
        %iter_i32 = index.castu %iter : index to i32
        %mul = arith.muli %cst_32, %iter_i32 : i32
        %idx_i32 = arith.addi %mul, %tx_i32 : i32
        %idx = index.castu %idx_i32 : i32 to index
        %elem = memref.load %input[%idx] : memref<4xf32>
        %gt = arith.cmpf ogt, %elem, %lane_max_iter : f32
        %lane_res_next = arith.select %gt, %idx_i32, %lane_res_iter : i32
        %lane_max_next = arith.select %gt, %elem, %lane_max_iter : f32
        scf.yield %lane_res_next, %lane_max_next : i32, f32
      }

      // CHECK: %[[SUBGROUP_MAX:.*]] = spirv.GroupNonUniformFMax <Subgroup> <Reduce> %[[LANE_MAX]] : f32 -> f32
      // CHECK: %[[OEQ:.*]] = spirv.FOrdEqual %[[LANE_MAX]], %[[SUBGROUP_MAX]] : f32
      // CHECK: %[[BALLOT:.*]] = spirv.GroupNonUniformBallot <Subgroup> %[[OEQ]] : vector<4xi32>
      // CHECK: %[[BALLOTLSB:.*]] = spirv.GroupNonUniformBallotFindLSB <Subgroup> %[[BALLOT]] : vector<4xi32>, i32
      // CHECK: %[[EQ:.*]] = spirv.IEqual %[[LOCALINVOCATIONIDX]], %[[C1]] : i32
      %subgroup_max = gpu.subgroup_reduce maximumf %lane_max : (f32) -> (f32)
      %eq = arith.cmpf oeq, %lane_max, %subgroup_max : f32
      %ballot = spirv.GroupNonUniformBallot <Subgroup> %eq : vector<4xi32>
      %lsb = spirv.GroupNonUniformBallotFindLSB <Subgroup> %ballot : vector<4xi32>, i32
      %cond = arith.cmpi eq, %cst_1_i32, %tx_i32 : i32

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
        memref.store %lane_res, %output[] : memref<i32>
      }

      // CHECK: spirv.Return
      gpu.return
    }
  }
}
