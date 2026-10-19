; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-expand-wide-atomics < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM

; 64- and 128-bit atomics take a lock from a table chosen by the address of
; the 16-byte block, and access the value with 32-bit atomics while they
; hold it, inside the loop that takes the lock.

; CHECK-LABEL: @add64(
; CHECK: lshr i64 {{.*}}, 4
; CHECK: and i64 {{.*}}, 1023
; CHECK: getelementptr inbounds [1024 x i32], ptr addrspace(1) @__agx_atomic_locks
; CHECK: wide.atomic:
; CHECK: cmpxchg ptr addrspace(1) %{{.*}}, i32 0, i32 1 monotonic monotonic
; CHECK: wide.atomic.locked:
; CHECK: fence seq_cst
; CHECK-COUNT-2: atomicrmw or ptr addrspace(1) %{{.*}}, i32 0 monotonic
; CHECK: add i64
; CHECK-COUNT-2: atomicrmw xchg ptr addrspace(1) %{{.*}}, i32 %{{.*}} monotonic
; CHECK: fence seq_cst
; CHECK: atomicrmw xchg ptr addrspace(1) %{{.*}}, i32 0 monotonic
; CHECK: wide.atomic.latch:
; ASM-LABEL: add64:
; ASM: global data buffer at
; ASM: atomic cmpxchg
define agx_kernel void @add64(ptr addrspace(1) %p, ptr addrspace(1) %o, i64 %v) {
  %a = atomicrmw add ptr addrspace(1) %p, i64 %v monotonic
  store i64 %a, ptr addrspace(1) %o
  ret void
}

; A compare and swap stores only when the comparison succeeds.
; CHECK-LABEL: @cas128(
; CHECK-COUNT-4: atomicrmw or
; CHECK: icmp eq i128
; CHECK: br i1 %{{.*}}, label %wide.atomic.store, label %wide.atomic.unlock
; CHECK: wide.atomic.store:
; CHECK-COUNT-4: atomicrmw xchg
define agx_kernel void @cas128(ptr addrspace(1) %p, ptr addrspace(1) %o, i128 %c, i128 %n) {
  %x = cmpxchg ptr addrspace(1) %p, i128 %c, i128 %n seq_cst seq_cst
  %v = extractvalue {i128, i1} %x, 0
  store i128 %v, ptr addrspace(1) %o
  ret void
}

; Atomic loads only read; double read-modify-write operations compute in
; software.
; CHECK-LABEL: @load_fadd(
; CHECK-COUNT-2: atomicrmw or
; CHECK-NOT: atomicrmw xchg ptr addrspace(1) %p
; CHECK: fadd double
define agx_kernel void @load_fadd(ptr addrspace(1) %p, ptr addrspace(1) %q, ptr addrspace(1) %o, double %d) {
  %l = load atomic i64, ptr addrspace(1) %p seq_cst, align 8
  store i64 %l, ptr addrspace(1) %o
  %f = atomicrmw fadd ptr addrspace(1) %q, double %d monotonic
  %o2 = getelementptr i64, ptr addrspace(1) %o, i32 1
  store double %f, ptr addrspace(1) %o2
  ret void
}
