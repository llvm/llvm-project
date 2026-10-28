// REQUIRES: host-supports-spirv
// RUN: mlir-opt %s --gpu-module-to-binary="format=isa" | FileCheck %s

// The nsw/nuw flags of integer ops reach the SPIR-V as NoSignedWrap and
// NoUnsignedWrap decorations. They need SPV_KHR_no_integer_wrap_decoration,
// which the SPIR-V backend declares only when a decoration uses it.

module attributes {gpu.container_module} {
  // CHECK-LABEL: gpu.binary @with_flags
  // CHECK-SAME: SPV_KHR_no_integer_wrap_decoration
  gpu.module @with_flags [#xevm.target] {
    llvm.func spir_kernelcc @add(%a: i64, %b: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %0 = llvm.add %a, %b overflow<nsw> : i64
      %1 = llvm.add %0, %b overflow<nuw> : i64
      llvm.store %1, %out : i64, !llvm.ptr<1>
      llvm.return
    }
  }

  // CHECK-LABEL: gpu.binary @without_flags
  // CHECK-NOT: SPV_KHR_no_integer_wrap_decoration
  gpu.module @without_flags [#xevm.target] {
    llvm.func spir_kernelcc @add(%a: i64, %b: i64, %out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %0 = llvm.add %a, %b : i64
      llvm.store %0, %out : i64, !llvm.ptr<1>
      llvm.return
    }
  }
}
