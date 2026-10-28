; In AIR the vote is over the active threads; Metal compares the mask of
; air.simd_vote_all with every thread of the SIMD-group (false with only lanes
; 0-15 active and a mask of 0xffff). It becomes a comparison with the ballot of
; true; air.simd_vote_any is kept.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

declare i64 @air.simd_ballot.i64(i1)
declare i1 @air.simd_vote_all.i64(i64)
declare i1 @air.simd_vote_any.i64(i64)
declare i1 @air.quad_vote_all(i16)

; CHECK-LABEL: define void @all
; CHECK: [[M:%.*]] = call i64 @air.simd_ballot.i64(i1 %p)
; CHECK: [[A:%.*]] = call i64 @air.simd_ballot.i64(i1 true)
; CHECK: [[V:%.*]] = icmp eq i64 [[M]], [[A]]
; CHECK: zext i1 [[V]] to i32
; CHECK: call i1 @air.simd_vote_any.i64(
; CHECK-NOT: simd_vote_all
define void @all(ptr addrspace(1) %in, ptr addrspace(1) %out) "air-kernel" {
  %x = load i32, ptr addrspace(1) %in
  %p = icmp ne i32 %x, 0
  %m = call i64 @air.simd_ballot.i64(i1 %p)
  %v = call i1 @air.simd_vote_all.i64(i64 %m)
  %r = zext i1 %v to i32
  store i32 %r, ptr addrspace(1) %out
  %w = call i1 @air.simd_vote_any.i64(i64 %m)
  %s = zext i1 %w to i32
  %o1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %s, ptr addrspace(1) %o1
  ret void
}

; CHECK-LABEL: define void @quad
; CHECK: [[A:%.*]] = call i16 @air.quad_ballot(i1 true)
; CHECK: icmp eq i16 %m, [[A]]
; CHECK-NOT: quad_vote_all
define void @quad(ptr addrspace(1) %in, ptr addrspace(1) %out) "air-kernel" {
  %x = load i16, ptr addrspace(1) %in
  %m = and i16 %x, 15
  %v = call i1 @air.quad_vote_all(i16 %m)
  %r = zext i1 %v to i32
  store i32 %r, ptr addrspace(1) %out
  ret void
}
