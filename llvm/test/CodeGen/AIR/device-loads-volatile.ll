; Device loads in a loop whose body also stores to the same pointer get
; marked volatile.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: load volatile i32, ptr addrspace(1)

define void @kernel(ptr addrspace(1) %p, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %v = load i32, ptr addrspace(1) %p
  %inc = add i32 %v, 1
  store i32 %inc, ptr addrspace(1) %p
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Spinlock: the lock %p gets volatile; the lock-guarded buffer %q becomes
; device_coherent (volatile alone does not bypass the AGX per-threadgroup cache).

declare { i32, i1 } @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), i32, i32)

; CHECK-LABEL: define void @kernel_cas
; CHECK: load volatile i32, ptr addrspace(1) %p
; CHECK: call i32 @air.load.device_coherent.i32.p1i32(ptr addrspace(1) %q)
; CHECK: store volatile i32
define void @kernel_cas(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %expected, i32 %desired) {
entry:
  %cas = call { i32, i1 } @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %p, i32 %expected, i32 %desired)
  %prev = extractvalue { i32, i1 } %cas, 0
  %v = load i32, ptr addrspace(1) %p
  %u = load i32, ptr addrspace(1) %q
  %sum = add i32 %v, %u
  store i32 %sum, ptr addrspace(1) %p
  ret void
}
