; AIRTGBarrierInsert puts a barrier between a threadgroup store and a later
; threadgroup load (read after write), which may read what another thread
; stored: within a block, and at the end of a block whose successor loads
; first. Without it, threads of one SIMD-group read what another SIMD-group
; had not stored yet (seen on an Apple M2 Max with 64-thread threadgroups).
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @same_block
; CHECK: store i32 %v, ptr addrspace(3)
; CHECK-NOT: load i32, ptr addrspace(3)
; CHECK: call void @air.wg.barrier(i32 2, i32 1)
; CHECK: load i32, ptr addrspace(3)

; CHECK-LABEL: define void @next_block
; CHECK: store i32 %v, ptr addrspace(3)
; CHECK-NEXT: call void @air.wg.barrier(i32 2, i32 1)
; CHECK-NEXT: br label %read
; CHECK: read:
; CHECK: load i32, ptr addrspace(3)

@tg = internal addrspace(3) global [64 x i32] undef, align 4
@tg2 = internal addrspace(3) global [64 x i32] undef, align 4

declare { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()

define void @same_block(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %lt = call { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()
  %lid = extractvalue { i32, i32, i32 } %lt, 0
  %ip = getelementptr i32, ptr addrspace(1) %in, i32 %lid
  %v = load i32, ptr addrspace(1) %ip
  %sp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %lid
  store i32 %v, ptr addrspace(3) %sp
  %rl = sub i32 63, %lid
  %rp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %rl
  %r = load i32, ptr addrspace(3) %rp
  %op = getelementptr i32, ptr addrspace(1) %out, i32 %lid
  store i32 %r, ptr addrspace(1) %op
  ret void
}

define void @next_block(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %lt = call { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()
  %lid = extractvalue { i32, i32, i32 } %lt, 0
  %ip = getelementptr i32, ptr addrspace(1) %in, i32 %lid
  %v = load i32, ptr addrspace(1) %ip
  %sp = getelementptr [64 x i32], ptr addrspace(3) @tg2, i32 0, i32 %lid
  store i32 %v, ptr addrspace(3) %sp
  br label %read

read:
  %rl = sub i32 63, %lid
  %rp = getelementptr [64 x i32], ptr addrspace(3) @tg2, i32 0, i32 %rl
  %r = load i32, ptr addrspace(3) %rp
  %op = getelementptr i32, ptr addrspace(1) %out, i32 %lid
  store i32 %r, ptr addrspace(1) %op
  ret void
}
