// RUN: mlir-opt -convert-scf-to-spirv %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

func.func @loop_kernel(%arg2 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg3 : memref<10xf32, #spirv.storage_class<StorageBuffer>>) {
  // CHECK: %[[LB:.*]] = spirv.Constant 4 : i32
  %lb = arith.constant 4 : index
  // CHECK: %[[UB:.*]] = spirv.Constant 42 : i32
  %ub = arith.constant 42 : index
  // CHECK: %[[STEP:.*]] = spirv.Constant 2 : i32
  %step = arith.constant 2 : index
  // CHECK:      spirv.mlir.loop {
  // CHECK-NEXT:   spirv.Branch ^[[HEADER:.*]](%[[LB]] : i32)
  // CHECK:      ^[[HEADER]](%[[INDVAR:.*]]: i32):
  // CHECK:        %[[CMP:.*]] = spirv.SLessThan %[[INDVAR]], %[[UB]] : i32
  // CHECK:        spirv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
  // CHECK:      ^[[BODY]]:
  // CHECK:        %[[ZERO1:.*]] = spirv.Constant 0 : i32
  // CHECK:        %[[OFFSET1:.*]] = spirv.Constant 0 : i32
  // CHECK:        %[[STRIDE1:.*]] = spirv.Constant 1 : i32
  // CHECK:        %[[UPDATE1:.*]] = spirv.IMul %[[STRIDE1]], %[[INDVAR]] : i32
  // CHECK:        %[[INDEX1:.*]] = spirv.IAdd %[[OFFSET1]], %[[UPDATE1]] : i32
  // CHECK:        spirv.AccessChain {{%.*}}{{\[}}%[[ZERO1]], %[[INDEX1]]{{\]}}
  // CHECK:        %[[ZERO2:.*]] = spirv.Constant 0 : i32
  // CHECK:        %[[OFFSET2:.*]] = spirv.Constant 0 : i32
  // CHECK:        %[[STRIDE2:.*]] = spirv.Constant 1 : i32
  // CHECK:        %[[UPDATE2:.*]] = spirv.IMul %[[STRIDE2]], %[[INDVAR]] : i32
  // CHECK:        %[[INDEX2:.*]] = spirv.IAdd %[[OFFSET2]], %[[UPDATE2]] : i32
  // CHECK:        spirv.AccessChain {{%.*}}[%[[ZERO2]], %[[INDEX2]]]
  // CHECK:        %[[INCREMENT:.*]] = spirv.IAdd %[[INDVAR]], %[[STEP]] : i32
  // CHECK:        spirv.Branch ^[[HEADER]](%[[INCREMENT]] : i32)
  // CHECK:      ^[[MERGE]]
  // CHECK:        spirv.mlir.merge
  // CHECK:      }
  scf.for %arg4 = %lb to %ub step %step {
    %1 = memref.load %arg2[%arg4] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
    memref.store %1, %arg3[%arg4] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  }
  return
}

// CHECK-LABEL: @loop_yield
func.func @loop_yield(%arg2 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg3 : memref<10xf32, #spirv.storage_class<StorageBuffer>>) {
  // CHECK: %[[LB:.*]] = spirv.Constant 4 : i32
  %lb = arith.constant 4 : index
  // CHECK: %[[UB:.*]] = spirv.Constant 42 : i32
  %ub = arith.constant 42 : index
  // CHECK: %[[STEP:.*]] = spirv.Constant 2 : i32
  %step = arith.constant 2 : index
  // CHECK: %[[INITVAR1:.*]] = spirv.Constant 0.000000e+00 : f32
  %s0 = arith.constant 0.0 : f32
  // CHECK: %[[INITVAR2:.*]] = spirv.Constant 1.000000e+00 : f32
  %s1 = arith.constant 1.0 : f32
  // CHECK: %[[VAR1:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: %[[VAR2:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
  // CHECK: spirv.mlir.loop {
  // CHECK:   spirv.Branch ^[[HEADER:.*]](%[[LB]], %[[INITVAR1]], %[[INITVAR2]] : i32, f32, f32)
  // CHECK: ^[[HEADER]](%[[INDVAR:.*]]: i32, %[[CARRIED1:.*]]: f32, %[[CARRIED2:.*]]: f32):
  // CHECK:   %[[CMP:.*]] = spirv.SLessThan %[[INDVAR]], %[[UB]] : i32
  // CHECK:   spirv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
  // CHECK: ^[[BODY]]:
  // CHECK:   %[[UPDATED:.*]] = spirv.FAdd %[[CARRIED1]], %[[CARRIED1]] : f32
  // CHECK-DAG:   %[[INCREMENT:.*]] = spirv.IAdd %[[INDVAR]], %[[STEP]] : i32
  // CHECK-DAG:   spirv.Store "Function" %[[VAR1]], %[[UPDATED]] : f32
  // CHECK-DAG:   spirv.Store "Function" %[[VAR2]], %[[UPDATED]] : f32
  // CHECK: spirv.Branch ^[[HEADER]](%[[INCREMENT]], %[[UPDATED]], %[[UPDATED]] : i32, f32, f32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spirv.mlir.merge
  // CHECK: }
  %result:2 = scf.for %i0 = %lb to %ub step %step iter_args(%si = %s0, %sj = %s1) -> (f32, f32) {
    %sn = arith.addf %si, %si : f32
    scf.yield %sn, %sn : f32, f32
  }
  // CHECK-DAG: %[[OUT1:.*]] = spirv.Load "Function" %[[VAR1]] : f32
  // CHECK-DAG: %[[OUT2:.*]] = spirv.Load "Function" %[[VAR2]] : f32
  // CHECK: spirv.Store "StorageBuffer" {{%.*}}, %[[OUT1]] : f32
  // CHECK: spirv.Store "StorageBuffer" {{%.*}}, %[[OUT2]] : f32
  memref.store %result#0, %arg3[%lb] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  memref.store %result#1, %arg3[%ub] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
  return
}

} // end module
