// RUN: mlir-opt -finalize-memref-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @atomic_rmw_f32
func.func @atomic_rmw_f32(%mem : memref<f32>, %val : f32) {
  // CHECK: %[[LOADED_F32:.*]] = llvm.load %{{.*}} : !llvm.ptr -> f32
  // CHECK: llvm.br ^[[LOOP:.*]](%[[LOADED_F32]] : f32)

  // CHECK: ^[[LOOP]](%[[ITER_F32:.*]]: f32):
  // CHECK-NEXT: %[[ITER_I32:.*]] = llvm.bitcast %[[ITER_F32]] : f32 to i32
  // CHECK-NEXT: %[[NEW_VAL_I32:.*]] = llvm.bitcast %{{.*}} : f32 to i32
  // CHECK-NEXT: %[[RES:.*]] = llvm.cmpxchg %{{.*}}, %[[ITER_I32]], %[[NEW_VAL_I32]] acq_rel monotonic : !llvm.ptr, i32
  // CHECK-NEXT: %[[NEW_LOADED_I32:.*]] = llvm.extractvalue %[[RES]][0]
  // CHECK-NEXT: %[[OK:.*]] = llvm.extractvalue %[[RES]][1]
  // CHECK-NEXT: %[[NEW_LOADED_F32:.*]] = llvm.bitcast %[[NEW_LOADED_I32]] : i32 to f32
  // CHECK-NEXT: llvm.cond_br %[[OK]], ^[[END:.*]], ^[[LOOP]](%[[NEW_LOADED_F32]] : f32)

  %x = memref.generic_atomic_rmw %mem[] : memref<f32> {
  ^bb0(%current_val: f32):
    memref.atomic_yield %val : f32
  }
  return
}
