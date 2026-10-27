; AIR's atomic functions become atomicrmw and cmpxchg with their memory order
; and scope (threadgroup memory and thread_scope_threadgroup are workgroup
; scope), and keep their volatile operand. An ordered LLVM atomic reaches AIR
; as a relaxed one between sequentially consistent fences; with the order it
; came from (agx.atomic.ordering, from AIR's portable passes) it is that
; atomic again, and without (@nohint) the fences stay. A compare-exchange
; leaves the value it found in its expected operand.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %t.air.ll | FileCheck %s
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %t.air.ll | FileCheck %s --check-prefix=NOHINT
; RUN: llc -mtriple=agx -mcpu=agx-g14g %t.air.ll -o - | FileCheck %s --check-prefix=ASM

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %[[A:a[0-9]*]] = atomicrmw add ptr addrspace(1) %u, i32 %{{tid_x|i}} monotonic
; CHECK-NOT: fence
; CHECK: %[[B:b[0-9]*]] = atomicrmw umax ptr addrspace(1) %u, i32 %[[A]] seq_cst
; CHECK-NOT: fence
; CHECK: atomicrmw fadd ptr addrspace(1) %f, float 1.000000e+00 monotonic
; CHECK: %[[D:d[0-9]*]] = atomicrmw min ptr addrspace(1) %u, i32 %[[B]] monotonic
; CHECK: %[[E:e[0-9]*]] = atomicrmw xchg ptr addrspace(1) %u, i32 %[[D]] monotonic
; CHECK: store i32 %[[E]], ptr addrspace(5) %[[EXP:.*]], align 4
; CHECK: %[[OLD:.*]] = load i32, ptr addrspace(5) %[[EXP]]
; CHECK: %[[X:.*]] = cmpxchg ptr addrspace(1) %u, i32 %[[OLD]], i32 5 monotonic monotonic
; CHECK: %[[V:.*]] = extractvalue { i32, i1 } %[[X]], 0
; CHECK: store i32 %[[V]], ptr addrspace(5) %[[EXP]]
; CHECK: atomicrmw add ptr addrspace(3) @tg, i32 %{{tid_x|i}} syncscope("workgroup") monotonic
; CHECK-NOT: fence
; CHECK: cmpxchg ptr addrspace(1) %u, i32 %{{.*}}, i32 7 acq_rel acquire
; CHECK-NOT: fence
; CHECK: atomicrmw add ptr addrspace(3) @tg, i32 1 syncscope("workgroup") release
; CHECK-NOT: fence
; CHECK: ret void

; NOHINT-LABEL: define agx_kernel void @nohint(
; NOHINT: fence seq_cst
; NOHINT-NEXT: atomicrmw volatile add ptr addrspace(1) %u, i32 1 monotonic
; NOHINT-NEXT: fence seq_cst

; ASM: atomic add,
; ASM: atomic umax,
; ASM: atomic cmpxchg,

@tg = internal addrspace(3) global [32 x i32] undef, align 4
declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @k(ptr addrspace(1) %f, ptr addrspace(1) %u) "air-kernel" {
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %i = extractvalue { i32, i32, i32 } %t, 0
  %a = atomicrmw add ptr addrspace(1) %u, i32 %i monotonic
  %b = atomicrmw umax ptr addrspace(1) %u, i32 %a seq_cst
  %c = atomicrmw fadd ptr addrspace(1) %f, float 1.0 monotonic
  %d = atomicrmw min ptr addrspace(1) %u, i32 %b monotonic
  %e = atomicrmw xchg ptr addrspace(1) %u, i32 %d monotonic
  %x = cmpxchg ptr addrspace(1) %u, i32 %e, i32 5 monotonic monotonic
  %ok = extractvalue { i32, i1 } %x, 1
  %s = select i1 %ok, i32 1, i32 2
  %g = atomicrmw add ptr addrspace(3) @tg, i32 %i monotonic
  %sum = add i32 %s, %g
  %y = cmpxchg ptr addrspace(1) %u, i32 %sum, i32 7 acq_rel acquire
  %h = atomicrmw add ptr addrspace(3) @tg, i32 1 release
  store i32 %h, ptr addrspace(1) %u
  ret void
}

declare void @air.atomic.fence(i32, i32, i32)
declare i32 @air.atomic.global.add.s.i32(ptr addrspace(1), i32, i32, i32, i1)

define void @nohint(ptr addrspace(1) %u) "air-kernel" {
  call void @air.atomic.fence(i32 3, i32 5, i32 2)
  %a = call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %u, i32 1, i32 0, i32 2, i1 true)
  call void @air.atomic.fence(i32 3, i32 5, i32 2)
  store i32 %a, ptr addrspace(1) %u
  ret void
}
