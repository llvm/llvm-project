; LLVM atomics become AIR atomic functions in their MSL 4.0 form: relaxed
; operations, with an ordered LLVM atomic between a release and an acquire
; air.atomic.fence. The scope is the threadgroup for threadgroup memory and
; workgroup-scoped atomics, the device otherwise. Unsigned min/max put the
; sign in the suffix (max.u), float operations have none.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: atomicrmw
; CHECK-NOT: cmpxchg ptr

; seq_cst on device memory: device-scope fences around a relaxed add.
; CHECK: call void @air.atomic.fence(i32 3, i32 5, i32 2)
; CHECK-NEXT: call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 2, i1 true)
; CHECK-NEXT: call void @air.atomic.fence(i32 3, i32 5, i32 2)

; seq_cst on threadgroup memory: threadgroup-scope fences.
; CHECK: call void @air.atomic.fence(i32 2, i32 5, i32 1)
; CHECK-NEXT: call i32 @air.atomic.local.xchg.i32(ptr addrspace(3) %l, i32 %v, i32 0, i32 1, i1 true)
; CHECK-NEXT: call void @air.atomic.fence(i32 2, i32 5, i32 1)

; Monotonic: no fences.
; CHECK-NEXT: call i32 @air.atomic.global.max.u.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 2, i1 true)
; CHECK-NEXT: call i32 @air.atomic.global.min.u.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 2, i1 true)
; CHECK-NEXT: call float @air.atomic.global.add.f32(ptr addrspace(1) %gf, float %f, i32 0, i32 2, i1 true)
; CHECK-NEXT: call i32 @air.atomic.global.max.s.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 1, i1 true)

; cmpxchg: one weak exchange through a private expected value.
; CHECK: store i32 %v, ptr %cmpxchg.expected
; CHECK-NEXT: %[[OLD:.*]] = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %g, ptr %cmpxchg.expected, i32 7, i32 0, i32 0, i32 2, i1 true)
; CHECK-NEXT: icmp eq i32 %[[OLD]], %v

; fence: singlethread fences are dropped.
; CHECK: call void @air.atomic.fence(i32 2, i32 5, i32 1)
; CHECK-NEXT: call void @air.atomic.fence(i32 3, i32 5, i32 2)
; CHECK-NEXT: ret void

define void @kernel(ptr addrspace(1) %g, ptr addrspace(3) %l, ptr addrspace(1) %gf, i32 %v, float %f, ptr addrspace(1) %out) {
entry:
  %a = atomicrmw add ptr addrspace(1) %g, i32 %v seq_cst
  %b = atomicrmw xchg ptr addrspace(3) %l, i32 %v seq_cst
  %c = atomicrmw umax ptr addrspace(1) %g, i32 %v monotonic
  %d = atomicrmw umin ptr addrspace(1) %g, i32 %v monotonic
  %e = atomicrmw fadd ptr addrspace(1) %gf, float %f monotonic
  %h = atomicrmw max ptr addrspace(1) %g, i32 %v syncscope("workgroup") monotonic
  %pair = cmpxchg ptr addrspace(1) %g, i32 %v, i32 7 monotonic monotonic
  %ok = extractvalue { i32, i1 } %pair, 1
  %okz = zext i1 %ok to i32
  store i32 %okz, ptr addrspace(1) %out
  fence syncscope("singlethread") seq_cst
  fence syncscope("workgroup") acq_rel
  fence seq_cst
  ret void
}
