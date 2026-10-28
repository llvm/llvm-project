; A threadgroup of at most 4 SIMD-groups ("agx-max-threads-per-threadgroup"
; at most 128) is one aligned block: its threadgroup barriers become block
; barriers, without the workgroup fences around them, where the subtarget has
; block_barrier (G14, not agx-strict). Larger or unknown threadgroups, other
; chips and strict mode keep the threadgroup barrier.
; RUN: llc -mtriple=agx -mcpu=agx-g14g < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g < %s | FileCheck %s --check-prefix=FULL
; RUN: llc -mtriple=agx -mcpu=agx-g14g -mattr=+agx-strict < %s | FileCheck %s --check-prefix=FULL

declare void @llvm.agx.threadgroup.barrier()

; CHECK-LABEL: small:
; CHECK: threadgroup_store
; CHECK-NEXT: block_barrier
; CHECK-NEXT: threadgroup_load
; FULL-LABEL: small:
; FULL: threadgroup_barrier
define agx_kernel void @small(ptr addrspace(3) %p, ptr addrspace(1) %q) "agx-max-threads-per-threadgroup"="128" {
  store i32 1, ptr addrspace(3) %p
  fence syncscope("workgroup") release
  call void @llvm.agx.threadgroup.barrier()
  fence syncscope("workgroup") acquire
  %v = load i32, ptr addrspace(3) %p
  store i32 %v, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: large:
; CHECK: threadgroup_barrier
; CHECK-NOT: block_barrier
define agx_kernel void @large(ptr addrspace(3) %p, ptr addrspace(1) %q) "agx-max-threads-per-threadgroup"="256" {
  store i32 1, ptr addrspace(3) %p
  fence syncscope("workgroup") seq_cst
  call void @llvm.agx.threadgroup.barrier()
  %v = load i32, ptr addrspace(3) %p
  store i32 %v, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: unknown:
; CHECK: threadgroup_barrier
; CHECK-NOT: block_barrier
define agx_kernel void @unknown(ptr addrspace(3) %p, ptr addrspace(1) %q) {
  store i32 1, ptr addrspace(3) %p
  call void @llvm.agx.threadgroup.barrier()
  %v = load i32, ptr addrspace(3) %p
  store i32 %v, ptr addrspace(1) %q
  ret void
}
