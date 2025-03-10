// RUN: mlir-opt -convert-spirv-to-llvm -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ControlBarrierOp
//===----------------------------------------------------------------------===//

// CHECK:           llvm.func spir_funccc @_Z22__spirv_ControlBarrieriii(i32, i32, i32) attributes {convergent, no_unwind, will_return}

// CHECK-LABEL: @control_barrier
spirv.func @control_barrier() "None" {
  // CHECK:         [[EXECUTION:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[MEMORY:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[SEMANTICS:%.*]] = llvm.mlir.constant(768 : i32) : i32
  // CHECK:         llvm.call spir_funccc @_Z22__spirv_ControlBarrieriii([[EXECUTION]], [[MEMORY]], [[SEMANTICS]]) {convergent, no_unwind, will_return} : (i32, i32, i32) -> ()
  spirv.ControlBarrier <Workgroup>, <Workgroup>, <CrossWorkgroupMemory|WorkgroupMemory>

  // CHECK:         [[EXECUTION:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[MEMORY:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[SEMANTICS:%.*]] = llvm.mlir.constant(256 : i32) : i32
  // CHECK:         llvm.call spir_funccc @_Z22__spirv_ControlBarrieriii([[EXECUTION]], [[MEMORY]], [[SEMANTICS]]) {convergent, no_unwind, will_return} : (i32, i32, i32) -> ()
  spirv.ControlBarrier <Workgroup>, <Workgroup>, <WorkgroupMemory>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.SplitBarrier
//===----------------------------------------------------------------------===//

// CHECK-DAG:           llvm.func spir_funccc @_Z33__spirv_ControlBarrierArriveINTELiii(i32, i32, i32) attributes {convergent, no_unwind, will_return}
// CHECK-DAG:           llvm.func spir_funccc @_Z31__spirv_ControlBarrierWaitINTELiii(i32, i32, i32) attributes {convergent, no_unwind, will_return}

// CHECK-LABEL: @split_barrier
spirv.func @split_barrier() "None" {
  // CHECK:         [[EXECUTION:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[MEMORY:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[SEMANTICS:%.*]] = llvm.mlir.constant(768 : i32) : i32
  // CHECK:         llvm.call spir_funccc @_Z33__spirv_ControlBarrierArriveINTELiii([[EXECUTION]], [[MEMORY]], [[SEMANTICS]]) {convergent, no_unwind, will_return} : (i32, i32, i32) -> ()
  spirv.INTEL.ControlBarrierArrive <Workgroup> <Workgroup> <CrossWorkgroupMemory|WorkgroupMemory>

  // CHECK:         [[EXECUTION:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[MEMORY:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:         [[SEMANTICS:%.*]] = llvm.mlir.constant(256 : i32) : i32
  // CHECK:         llvm.call spir_funccc @_Z31__spirv_ControlBarrierWaitINTELiii([[EXECUTION]], [[MEMORY]], [[SEMANTICS]]) {convergent, no_unwind, will_return} : (i32, i32, i32) -> ()
  spirv.INTEL.ControlBarrierWait <Workgroup> <Workgroup> <WorkgroupMemory>
  spirv.Return
}
