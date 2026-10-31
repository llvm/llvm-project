; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -stop-after=agx-lower-generic-addresses < %s | FileCheck %s --check-prefix=CALLS
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; With real calls, a called function can be given a generic pointer to its
; caller's private memory (Blender Cycles' BVH traversal writes the kernel's
; Intersection through one). The callee has no cast of its own, yet its
; generic accesses must test the private tag (bit 62) the module makes.

; CALLS-LABEL: define agx_kernel void @k(
; CALLS: or i64 %{{.*}}, 4611686018427387904
; CALLS-LABEL: define internal void @fill(
; CALLS: lshr i64 %{{.*}}, 32
; CALLS: lshr i32 %{{.*}}, 30
; CALLS: icmp eq i32 %{{.*}}, 1
; CALLS: store i32 %{{.*}}, ptr addrspace(5)
; CALLS: store i32 %{{.*}}, ptr addrspace(1)
; ASM-LABEL: k:
; ASM: call fill
; ASM-LABEL: fill:
; ASM-DAG: stack_store
; ASM-DAG: device_store

define agx_kernel void @k(ptr addrspace(1) %out, i32 %v) {
  %buf = alloca i32, align 4, addrspace(5)
  %g = addrspacecast ptr addrspace(5) %buf to ptr
  call void @fill(ptr %g, i32 %v)
  %r = load i32, ptr addrspace(5) %buf
  store i32 %r, ptr addrspace(1) %out
  ret void
}

define internal void @fill(ptr %p, i32 %v) noinline {
  store i32 %v, ptr %p
  ret void
}
