// RUN: mlir-opt -convert-scf-to-spirv %s -o - | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {

func.func @loop_kernel(%arg2 : memref<10xf32>, %arg3 : memref<10xf32>) {
  // CHECK: %[[LB:.*]] = spv.Constant 4 : i32
  %lb = arith.constant 4 : index
  // CHECK: %[[UB:.*]] = spv.Constant 42 : i32
  %ub = arith.constant 42 : index
  // CHECK: %[[STEP:.*]] = spv.Constant 2 : i32
  %step = arith.constant 2 : index
  // CHECK:      spv.mlir.loop {
  // CHECK-NEXT:   spv.Branch ^[[HEADER:.*]](%[[LB]] : i32)
  // CHECK:      ^[[HEADER]](%[[INDVAR:.*]]: i32):
  // CHECK:        %[[CMP:.*]] = spv.SLessThan %[[INDVAR]], %[[UB]] : i32
  // CHECK:        spv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
  // CHECK:      ^[[BODY]]:
  // CHECK:        %[[ZERO1:.*]] = spv.Constant 0 : i32
  // CHECK:        %[[OFFSET1:.*]] = spv.Constant 0 : i32
  // CHECK:        %[[STRIDE1:.*]] = spv.Constant 1 : i32
  // CHECK:        %[[UPDATE1:.*]] = spv.IMul %[[STRIDE1]], %[[INDVAR]] : i32
  // CHECK:        %[[INDEX1:.*]] = spv.IAdd %[[OFFSET1]], %[[UPDATE1]] : i32
  // CHECK:        spv.AccessChain {{%.*}}{{\[}}%[[ZERO1]], %[[INDEX1]]{{\]}}
  // CHECK:        %[[ZERO2:.*]] = spv.Constant 0 : i32
  // CHECK:        %[[OFFSET2:.*]] = spv.Constant 0 : i32
  // CHECK:        %[[STRIDE2:.*]] = spv.Constant 1 : i32
  // CHECK:        %[[UPDATE2:.*]] = spv.IMul %[[STRIDE2]], %[[INDVAR]] : i32
  // CHECK:        %[[INDEX2:.*]] = spv.IAdd %[[OFFSET2]], %[[UPDATE2]] : i32
  // CHECK:        spv.AccessChain {{%.*}}[%[[ZERO2]], %[[INDEX2]]]
  // CHECK:        %[[INCREMENT:.*]] = spv.IAdd %[[INDVAR]], %[[STEP]] : i32
  // CHECK:        spv.Branch ^[[HEADER]](%[[INCREMENT]] : i32)
  // CHECK:      ^[[MERGE]]
  // CHECK:        spv.mlir.merge
  // CHECK:      }
  scf.for %arg4 = %lb to %ub step %step {
    %1 = memref.load %arg2[%arg4] : memref<10xf32>
    memref.store %1, %arg3[%arg4] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: @loop_yield
func.func @loop_yield(%arg2 : memref<10xf32>, %arg3 : memref<10xf32>) {
  // CHECK: %[[LB:.*]] = spv.Constant 4 : i32
  %lb = arith.constant 4 : index
  // CHECK: %[[UB:.*]] = spv.Constant 42 : i32
  %ub = arith.constant 42 : index
  // CHECK: %[[STEP:.*]] = spv.Constant 2 : i32
  %step = arith.constant 2 : index
  // CHECK: %[[INITVAR1:.*]] = spv.Constant 0.000000e+00 : f32
  %s0 = arith.constant 0.0 : f32
  // CHECK: %[[INITVAR2:.*]] = spv.Constant 1.000000e+00 : f32
  %s1 = arith.constant 1.0 : f32
  // CHECK: %[[VAR1:.*]] = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: %[[VAR2:.*]] = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.mlir.loop {
  // CHECK:   spv.Branch ^[[HEADER:.*]](%[[LB]], %[[INITVAR1]], %[[INITVAR2]] : i32, f32, f32)
  // CHECK: ^[[HEADER]](%[[INDVAR:.*]]: i32, %[[CARRIED1:.*]]: f32, %[[CARRIED2:.*]]: f32):
  // CHECK:   %[[CMP:.*]] = spv.SLessThan %[[INDVAR]], %[[UB]] : i32
  // CHECK:   spv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
  // CHECK: ^[[BODY]]:
  // CHECK:   %[[UPDATED:.*]] = spv.FAdd %[[CARRIED1]], %[[CARRIED1]] : f32
  // CHECK-DAG:   %[[INCREMENT:.*]] = spv.IAdd %[[INDVAR]], %[[STEP]] : i32
  // CHECK-DAG:   spv.Store "Function" %[[VAR1]], %[[UPDATED]] : f32
  // CHECK-DAG:   spv.Store "Function" %[[VAR2]], %[[UPDATED]] : f32
  // CHECK: spv.Branch ^[[HEADER]](%[[INCREMENT]], %[[UPDATED]], %[[UPDATED]] : i32, f32, f32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spv.mlir.merge
  // CHECK: }
  %result:2 = scf.for %i0 = %lb to %ub step %step iter_args(%si = %s0, %sj = %s1) -> (f32, f32) {
    %sn = arith.addf %si, %si : f32
    scf.yield %sn, %sn : f32, f32
  }
  // CHECK-DAG: %[[OUT1:.*]] = spv.Load "Function" %[[VAR1]] : f32
  // CHECK-DAG: %[[OUT2:.*]] = spv.Load "Function" %[[VAR2]] : f32
  // CHECK: spv.Store "StorageBuffer" {{%.*}}, %[[OUT1]] : f32
  // CHECK: spv.Store "StorageBuffer" {{%.*}}, %[[OUT2]] : f32
  memref.store %result#0, %arg3[%lb] : memref<10xf32>
  memref.store %result#1, %arg3[%ub] : memref<10xf32>
  return
}

} // end module
