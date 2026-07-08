// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test the device-side lowering of `declare target ... indirect` functions. A
// new global holding the address of the function is generated with protected
// visibility so the runtime can access it. A function marked `indirect = false`
// must not generate such a global.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK: @[[ENTRY:__omp_offloading_[0-9a-z]+_[0-9a-z]+_indirect_fn_l[0-9]+]] = protected constant ptr @indirect_fn
  llvm.func @indirect_fn() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter), automap = false, indirect = true>} {
    llvm.return
  }

  // A function marked `indirect = false` must not produce an indirect global.
  // CHECK-NOT: plain_fn_l
  llvm.func @plain_fn() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter), automap = false, indirect = false>} {
    llvm.return
  }
}
