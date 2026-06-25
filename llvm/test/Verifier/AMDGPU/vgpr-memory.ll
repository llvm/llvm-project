; RUN: not llvm-as %s --disable-output 2>&1 | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

; A "VGPR as memory" global is register-backed: it has no defined initial
; contents and per-lane storage, so it cannot be statically initialized or
; atomically accessed. An addrspacecast to/from addrspace(13) is allowed but
; lowers to poison (it has no meaningful numeric address), and likewise
; ptrtoint/inttoptr are allowed, so neither is diagnosed here.

; CHECK: atomic operations on the VGPR address space (13) are not allowed
; CHECK-NEXT: atomicrmw add ptr addrspace(13) @valid.poison
; CHECK: atomic operations on the VGPR address space (13) are not allowed
; CHECK-NEXT: %v = load atomic i32, ptr addrspace(13) @valid.poison
; CHECK: intrinsic with a VGPR address space (13) pointer argument is not allowed
; CHECK-NEXT: call void @llvm.memcpy
; CHECK: intrinsic with a VGPR address space (13) pointer argument is not allowed
; CHECK-NEXT: call void @llvm.memset
; CHECK: global variable in the VGPR address space (13) cannot have an initializer
; CHECK-NEXT: ptr addrspace(13) @bad.init
; CHECK: global variable in the VGPR address space (13) cannot have an initializer
; CHECK-NEXT: ptr addrspace(13) @bad.zeroinit

; A poison initializer (or none) is fine.
@valid.poison = internal addrspace(13) global i32 poison
@valid.array = internal addrspace(13) global [4 x i32] poison

@bad.init = internal addrspace(13) global i32 7
@bad.zeroinit = internal addrspace(13) global [2 x i32] zeroinitializer

define void @atomic_rmw() {
  atomicrmw add ptr addrspace(13) @valid.poison, i32 1 seq_cst
  ret void
}

define i32 @atomic_load() {
  %v = load atomic i32, ptr addrspace(13) @valid.poison seq_cst, align 4
  ret i32 %v
}

define void @memcpy_vgpr(ptr %src) {
  call void @llvm.memcpy.p13.p0.i64(ptr addrspace(13) @valid.poison, ptr %src, i64 16, i1 false)
  ret void
}

define void @memset_vgpr() {
  call void @llvm.memset.p13.i64(ptr addrspace(13) @valid.poison, i8 0, i64 16, i1 false)
  ret void
}
