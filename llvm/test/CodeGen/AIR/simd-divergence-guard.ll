; Metal's compiler if-converts a branch whose sides have no side effects, so a
; SIMD-group function in it runs with every thread of the SIMD-group active
; (air.simd_ballot in each side of `if (tid < 16)` gave 0xffffffff). The Metal
; pipeline puts llvm.sideeffect before the first air.simd_* / air.quad_* call of
; each block that runs under divergent control flow (does not post-dominate the
; entry, or is on a cycle); straight-line blocks are left alone. The portable
; AIR is unchanged.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE
; RUN: opt -mtriple=air -passes=air-simd-divergence-guard -S %s | FileCheck %s --check-prefix=OPT

; PORTABLE-NOT: llvm.sideeffect

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
declare i64 @air.simd_ballot.i64(i1)
declare i32 @air.simd_sum.u.i32(i32)
declare i32 @air.quad_shuffle_xor.u.i32(i32, i16)
declare i32 @air.simd_shuffle_xor.u.i32(i32, i16)

; CHECK-LABEL: define void @if_else
; CHECK: entry:
; CHECK-NOT: llvm.sideeffect
; CHECK: call i32 @air.simd_sum.u.i32
; CHECK: a:
; CHECK-NEXT: call void @llvm.sideeffect()
; CHECK-NEXT: call i64 @air.simd_ballot.i64(i1 true)
; CHECK-NOT: llvm.sideeffect
; CHECK: b:
; CHECK-NEXT: call void @llvm.sideeffect()
; CHECK-NEXT: call i64 @air.simd_ballot.i64(i1 true)
; CHECK: j:
; CHECK-NOT: llvm.sideeffect
; CHECK: call i32 @air.quad_shuffle_xor.u.i32
; CHECK: ret void
; OPT-LABEL: define void @if_else
; OPT-COUNT-2: call void @llvm.sideeffect()
; OPT-NOT: llvm.sideeffect
; OPT: ret void
define void @if_else(ptr addrspace(1) %o) "air-kernel" {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tid = extractvalue { i32, i32, i32 } %t, 0
  %s = call i32 @air.simd_sum.u.i32(i32 %tid)
  %lo = icmp ult i32 %tid, 16
  br i1 %lo, label %a, label %b
a:
  %ma = call i64 @air.simd_ballot.i64(i1 true)
  %ma2 = call i64 @air.simd_ballot.i64(i1 false)
  %xa = xor i64 %ma, %ma2
  br label %j
b:
  %mb = call i64 @air.simd_ballot.i64(i1 true)
  br label %j
j:
  %m = phi i64 [ %xa, %a ], [ %mb, %b ]
  %m32 = trunc i64 %m to i32
  %q = call i32 @air.quad_shuffle_xor.u.i32(i32 %m32, i16 1)
  %r = add i32 %q, %s
  %p = getelementptr i32, ptr addrspace(1) %o, i32 %tid
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; A loop body runs with fewer threads in later iterations.
; CHECK-LABEL: define void @loop
; CHECK: lp:
; CHECK: call void @llvm.sideeffect()
; CHECK-NEXT: call i32 @air.simd_shuffle_xor.u.i32
define void @loop(ptr addrspace(1) %o) "air-kernel" {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tid = extractvalue { i32, i32, i32 } %t, 0
  br label %lp
lp:
  %k = phi i32 [ 0, %entry ], [ %k1, %lp ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %lp ]
  %v = call i32 @air.simd_shuffle_xor.u.i32(i32 %acc, i16 1)
  %acc1 = add i32 %v, %k
  %k1 = add i32 %k, 1
  %more = icmp ult i32 %k1, %tid
  br i1 %more, label %lp, label %x
x:
  %p = getelementptr i32, ptr addrspace(1) %o, i32 %tid
  store i32 %acc1, ptr addrspace(1) %p
  ret void
}

; CHECK: declare void @llvm.sideeffect()
