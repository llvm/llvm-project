; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-threadgroup-address-offsets=false < %s | FileCheck %s --check-prefix=OFF
; RUN: opt -mtriple=agx -mcpu=agx-g14c -passes=loop-unroll -S < %s | FileCheck %s --check-prefix=UNROLL

@s = addrspace(3) global [1024 x float] poison, align 16

; PTX computes threadgroup addresses in integers: a[i + k] and a[i + k + 1]
; with k a multiple of 2 are two inttoptr of unrelated-looking expressions.
; Written as one base and constant offsets, they are one xy load.
; CHECK-LABEL: pair:
; CHECK:         threadgroup_load i32, xy,
; OFF-LABEL:   pair:
; OFF-COUNT-2:   threadgroup_load i32, x,
define agx_kernel void @pair(ptr addrspace(1) %out, i32 %i, i32 %k) {
  %k2 = shl i32 %k, 1
  %a = add i32 %k2, %i
  %b = or disjoint i32 %k2, 1
  %c = add i32 %b, %i
  %as = shl i32 %a, 2
  %cs = shl i32 %c, 2
  %ab = add i32 %as, ptrtoint (ptr addrspace(3) @s to i32)
  %cb = add i32 %cs, ptrtoint (ptr addrspace(3) @s to i32)
  %pa = inttoptr i32 %ab to ptr addrspace(3)
  %pc = inttoptr i32 %cb to ptr addrspace(3)
  %x = load float, ptr addrspace(3) %pa, align 8
  %y = load float, ptr addrspace(3) %pc, align 4
  %r = fadd float %x, %y
  store float %r, ptr addrspace(1) %out
  ret void
}

; A loop that reads 8 contiguous threadgroup bytes per iteration (two adjacent
; floats, the address advancing by 8) with a known trip count is unrolled
; twice: the copies' loads become one xyzw load.
; UNROLL-LABEL: @rows(
; UNROLL:         %i.next.1 = add nuw nsw i32 %i, 4
; UNROLL:         icmp eq i32 %i.next.1, 512
define agx_kernel void @rows(ptr addrspace(1) %out, i32 %row) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc2, %loop ]
  %idx = add i32 %i, %row
  %off = shl i32 %idx, 2
  %addr = add i32 %off, ptrtoint (ptr addrspace(3) @s to i32)
  %p = inttoptr i32 %addr to ptr addrspace(3)
  %x = load float, ptr addrspace(3) %p, align 4
  %q = getelementptr i8, ptr addrspace(3) %p, i32 4
  %y = load float, ptr addrspace(3) %q, align 4
  %s1 = fadd float %acc, %x
  %acc2 = fadd float %s1, %y
  %i.next = add nuw nsw i32 %i, 2
  %done = icmp eq i32 %i.next, 512
  br i1 %done, label %exit, label %loop
exit:
  store float %acc2, ptr addrspace(1) %out
  ret void
}

; The constant part of an address folded into a constant expression with the
; variable's address (as optimizations of portable AIR leave it): the two
; loads are still one base plus 0 and 4, and one xy load.
; CHECK-LABEL: const_expr_offset:
; CHECK: threadgroup_load i32, xy,
; OFF-LABEL: const_expr_offset:
; OFF-COUNT-2: threadgroup_load i32, x,
define void @const_expr_offset(ptr addrspace(1) %out, i32 %t) {
  %base = shl i32 %t, 4
  %a0 = add i32 %base, ptrtoint (ptr addrspace(3) @s to i32)
  %p0 = inttoptr i32 %a0 to ptr addrspace(3)
  %x = load i32, ptr addrspace(3) %p0, align 4
  %a1 = add i32 %base, add (i32 ptrtoint (ptr addrspace(3) @s to i32), i32 4)
  %p1 = inttoptr i32 %a1 to ptr addrspace(3)
  %y = load i32, ptr addrspace(3) %p1, align 4
  %sum = add i32 %x, %y
  store i32 %sum, ptr addrspace(1) %out
  ret void
}
