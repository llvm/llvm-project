; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; The llvm.amdgcn.addrspacecast.nonnull intrinsic is replaced by an
; addrspacecast instruction carrying the nonnull flag.

define ptr @local_to_flat(ptr addrspace(3) %ptr) {
  ; CHECK: %res = addrspacecast nonnull ptr addrspace(3) %ptr to ptr
  %res = call ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p3(ptr addrspace(3) %ptr)
  ret ptr %res
}

define ptr @private_to_flat(ptr addrspace(5) %ptr) {
  ; CHECK: %res = addrspacecast nonnull ptr addrspace(5) %ptr to ptr
  %res = call ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p5(ptr addrspace(5) %ptr)
  ret ptr %res
}

define ptr addrspace(3) @flat_to_local(ptr %ptr) {
  ; CHECK: %res = addrspacecast nonnull ptr %ptr to ptr addrspace(3)
  %res = call ptr addrspace(3) @llvm.amdgcn.addrspacecast.nonnull.p3.p0(ptr %ptr)
  ret ptr addrspace(3) %res
}

define ptr addrspace(5) @flat_to_private(ptr %ptr) {
  ; CHECK: %res = addrspacecast nonnull ptr %ptr to ptr addrspace(5)
  %res = call ptr addrspace(5) @llvm.amdgcn.addrspacecast.nonnull.p5.p0(ptr %ptr)
  ret ptr addrspace(5) %res
}

; A malformed call with too few arguments is dropped instead of upgraded.
define void @malformed_no_args() {
  ; CHECK-LABEL: @malformed_no_args(
  ; CHECK-NEXT: ret void
  call ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p3.malformed()
  ret void
}

declare ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p3(ptr addrspace(3))
declare ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p5(ptr addrspace(5))
declare ptr addrspace(3) @llvm.amdgcn.addrspacecast.nonnull.p3.p0(ptr)
declare ptr addrspace(5) @llvm.amdgcn.addrspacecast.nonnull.p5.p0(ptr)
declare ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p3.malformed()
