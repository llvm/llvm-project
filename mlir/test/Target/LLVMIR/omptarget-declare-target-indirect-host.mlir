// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test the host-side lowering of `declare target ... indirect` functions. The
// runtime needs an offload entry (flag 8 == OMPTargetGlobalVarEntryIndirect)
// so that indirect calls within a target region can be resolved. A function
// marked `indirect = false` must not register an offload entry.

// CHECK-DAG: %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_target_device = false} {
  // CHECK: @.offloading.entry.[[ENTRY:__omp_offloading_[0-9a-z]+_[0-9a-z]+_indirect_fn_l[0-9]+]] = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 8, ptr @indirect_fn, ptr @{{.*}}, i64 8, i64 0, ptr null }
  llvm.func @indirect_fn() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter), indirect = true>} {
    llvm.return
  }

  // A function marked `indirect = false` must not produce an offload entry.
  // CHECK-NOT: plain_fn_l
  llvm.func @plain_fn() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>} {
    llvm.return
  }
}
