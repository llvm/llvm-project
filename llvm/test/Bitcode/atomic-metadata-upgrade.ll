; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define void @foo(ptr %p, float %v) {
  ; CHECK: atomicrmw fadd ptr %p, float %v seq_cst, align 4, !atomic.ignore.denormal.mode !0
  %r = atomicrmw fadd ptr %p, float %v seq_cst, !amdgpu.ignore.denormal.mode !0
  ret void
}

!0 = !{}
