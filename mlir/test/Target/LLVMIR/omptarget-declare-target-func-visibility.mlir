// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Device-side `declare target` functions are externally visible by default so
// that they can be referenced from other device translation units. They are
// emitted with hidden visibility so that the offload LTO can internalize and
// delete them when they turn out to be unused in the final device image. This
// avoids leaving dead functions behind that may, for example, reference LDS and
// trigger spurious backend diagnostics.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK: define hidden void @device_any()
  llvm.func @device_any() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    llvm.return
  }

  // CHECK: define hidden void @device_nohost()
  llvm.func @device_nohost() attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
    llvm.return
  }

  // A function with an explicitly requested (non-default) visibility is left
  // untouched.
  // CHECK: define protected void @device_protected()
  llvm.func protected @device_protected() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    llvm.return
  }

  // A function that is not declare target is unaffected.
  // CHECK: define void @not_declare_target()
  llvm.func @not_declare_target() {
    llvm.return
  }

  // A declaration (no definition) is left untouched: there is nothing to
  // internalize, and hiding it could over-constrain the symbol's visibility.
  // CHECK: declare void @device_decl()
  llvm.func @device_decl() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>}
}
