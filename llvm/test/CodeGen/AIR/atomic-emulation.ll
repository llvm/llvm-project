; Metal's atomics are 32-bit. In device memory the others are emulated: 8-
; and 16-bit as a compare-exchange loop on the containing word, 64- and
; 128-bit under a lock of the table the kernel gets as a buffer parameter
; ("air_atomic_locks", location index 30), taken and released in one loop
; iteration; 32-bit types other than i32 and float as a loop on the bits.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @v2f16(
; CHECK: atomicrmw.loop:
; CHECK: fadd <2 x half>
; CHECK: call i32 @air.atomic.global.cmpxchg.weak.i32(
define void @v2f16(ptr addrspace(1) %p, ptr addrspace(1) %out, <2 x half> %v) "air-kernel" {
  %o = atomicrmw fadd ptr addrspace(1) %p, <2 x half> %v monotonic
  store <2 x half> %o, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: define void @fadd128(
; CHECK: atomiclock.held:
; CHECK-COUNT-4: call i32 @air.atomic.global.or.s.i32(
; CHECK: fadd <4 x float>
; CHECK-COUNT-4: call i32 @air.atomic.global.xchg.i32(
define void @fadd128(ptr addrspace(1) %p, ptr addrspace(1) %out, <4 x float> %v) "air-kernel" {
  %o = atomicrmw fadd ptr addrspace(1) %p, <4 x float> %v monotonic
  store <4 x float> %o, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: define void @fadd16(
; CHECK-NOT: air_atomic_locks
; CHECK: and i64 {{%.*}}, -4
; CHECK: atomic16.loop:
; CHECK: fadd half
; CHECK: call i32 @air.atomic.global.cmpxchg.weak.i32(
define void @fadd16(ptr addrspace(1) %p, ptr addrspace(1) %out, half %v) "air-kernel" {
  %o = atomicrmw fadd ptr addrspace(1) %p, half %v monotonic
  store half %o, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: define void @add64(
; CHECK-SAME: ptr addrspace(1) {{.*}}%air_atomic_locks,
; CHECK: [[A:%.*]] = ptrtoint ptr addrspace(1) %p to i64
; CHECK: [[S:%.*]] = lshr i64 [[A]], 3
; CHECK: and i32 {{.*}}, 1023
; CHECK: atomiclock.loop:
; CHECK: call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) [[LOCK:%[0-9]+]], ptr {{%.*}}, i32 1,
; CHECK: br i1 {{%.*}}, label %atomiclock.held, label %atomiclock.latch
; CHECK: atomiclock.held:
; CHECK: call i32 @air.atomic.global.or.s.i32(
; CHECK: call i32 @air.atomic.global.or.s.i32(
; CHECK: add i64
; CHECK: call i32 @air.atomic.global.xchg.i32(
; CHECK: call i32 @air.atomic.global.xchg.i32(
; CHECK: call i32 @air.atomic.global.xchg.i32(ptr addrspace(1) [[LOCK]], i32 0,
; CHECK: br label %atomiclock.latch
; CHECK: atomiclock.latch:
; CHECK: [[DONE:%.*]] = phi i1 [ false, %atomiclock.loop ], [ true, %atomiclock.held ]
; CHECK: br i1 [[DONE]], label {{%.*}}, label %atomiclock.loop
define void @add64(ptr addrspace(1) %p, ptr addrspace(1) %out, i64 %v) "air-kernel" {
  %o = atomicrmw add ptr addrspace(1) %p, i64 %v monotonic
  store i64 %o, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: define void @cmpxchg64(
; CHECK: atomiclock.held:
; CHECK-COUNT-2: call i32 @air.atomic.global.or.s.i32(
; CHECK: icmp eq i64
; CHECK: select i1
; CHECK-COUNT-2: call i32 @air.atomic.global.xchg.i32(
define void @cmpxchg64(ptr addrspace(1) %p, ptr addrspace(1) %out, i64 %c, i64 %n) "air-kernel" {
  %x = cmpxchg ptr addrspace(1) %p, i64 %c, i64 %n seq_cst seq_cst
  %o = extractvalue { i64, i1 } %x, 0
  store i64 %o, ptr addrspace(1) %out
  ret void
}

; The lock table's location index.
; CHECK: !"air.location_index", i32 30, i32 1, !"air.read_write", !"air.address_space", i32 1, {{.*}}!"air.arg_name", !"air_atomic_locks"}
