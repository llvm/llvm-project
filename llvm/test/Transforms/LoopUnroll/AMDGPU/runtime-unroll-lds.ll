; Runtime unrolling of loops that access LDS is governed by the LDS check and
; -amdgpu-unroll-runtime-local, whatever -amdgpu-unroll-runtime-default says,
; as before runtime unrolling was enabled by default: a loop the check admits
; is runtime unrolled iff -amdgpu-unroll-runtime-local is set, and a loop it
; rejects or never admits is not runtime unrolled. Loops without LDS follow the
; default.

; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll -S %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DEF-ON,LDS-ON
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll -S %s \
; RUN:   -amdgpu-unroll-runtime-local=false \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DEF-ON,LDS-OFF
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll -S %s \
; RUN:   -amdgpu-unroll-runtime-default=false \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DEF-OFF,LDS-ON
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=loop-unroll -S %s \
; RUN:   -amdgpu-unroll-runtime-default=false -amdgpu-unroll-runtime-local=false \
; RUN:   | FileCheck %s --check-prefixes=CHECK,DEF-OFF,LDS-OFF

@lds = internal addrspace(3) global [1024 x float] poison, align 4

; Admitted: depth 1, the LDS GEP addresses a global directly.
; CHECK-LABEL: @lds_admitted(
; LDS-ON:      for.body.epil:
; LDS-OFF-NOT: for.body.epil:
; CHECK:       ret void
define void @lds_admitted(ptr addrspace(1) %out, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %acc = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %idx = zext i32 %iv to i64
  %ptr = getelementptr inbounds [1024 x float], ptr addrspace(3) @lds, i64 0, i64 %idx
  %val = load float, ptr addrspace(3) %ptr, align 4
  %add = fadd float %acc, %val
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  %res = phi float [ 0.0, %entry ], [ %add, %for.body ]
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}

; Rejected: the LDS GEP's pointer operand is a derived pointer, not a global
; or an argument. Never runtime unrolled.
; CHECK-LABEL: @lds_derived_ptr(
; CHECK-NOT:   for.body.epil:
; CHECK:       ret void
define void @lds_derived_ptr(ptr addrspace(1) %out, i32 %n, i32 %base_off) {
entry:
  %base = getelementptr inbounds [1024 x float], ptr addrspace(3) @lds, i64 0, i32 %base_off
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %acc = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %ptr = getelementptr inbounds float, ptr addrspace(3) %base, i32 %iv
  %val = load float, ptr addrspace(3) %ptr, align 4
  %add = fadd float %acc, %val
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  %res = phi float [ 0.0, %entry ], [ %add, %for.body ]
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}

; Rejected: the LDS loop is nested at depth 3. Never runtime unrolled. The two
; enclosing loops are marked unroll.disable so that unrolling them cannot clone
; an inner.epil block into this function.
; CHECK-LABEL: @lds_deep(
; CHECK-NOT:   inner.epil:
; CHECK:       ret void
define void @lds_deep(ptr addrspace(1) %out, i32 %n1, i32 %n2, i32 %n3) {
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  %acc.o = phi float [ 0.0, %entry ], [ %acc.m.out, %outer.latch ]
  br label %mid

mid:
  %j = phi i32 [ 0, %outer ], [ %j.next, %mid.latch ]
  %acc.m = phi float [ %acc.o, %outer ], [ %acc.i.out, %mid.latch ]
  br label %inner

inner:
  %k = phi i32 [ 0, %mid ], [ %k.next, %inner ]
  %acc.i = phi float [ %acc.m, %mid ], [ %add, %inner ]
  %sum = add i32 %i, %j
  %sum2 = add i32 %sum, %k
  %off = and i32 %sum2, 1023
  %idx = zext i32 %off to i64
  %ptr = getelementptr inbounds [1024 x float], ptr addrspace(3) @lds, i64 0, i64 %idx
  %val = load float, ptr addrspace(3) %ptr, align 4
  %add = fadd float %acc.i, %val
  %k.next = add nuw nsw i32 %k, 1
  %cond.i = icmp eq i32 %k.next, %n3
  br i1 %cond.i, label %mid.latch, label %inner

mid.latch:
  %acc.i.out = phi float [ %add, %inner ]
  %j.next = add nuw nsw i32 %j, 1
  %cond.m = icmp eq i32 %j.next, %n2
  br i1 %cond.m, label %outer.latch, label %mid, !llvm.loop !0

outer.latch:
  %acc.m.out = phi float [ %acc.i.out, %mid.latch ]
  %i.next = add nuw nsw i32 %i, 1
  %cond.o = icmp eq i32 %i.next, %n1
  br i1 %cond.o, label %exit, label %outer, !llvm.loop !2

exit:
  %res = phi float [ %acc.m.out, %outer.latch ]
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}

; Never checked: amdgpu.loop.unroll.threshold lowers the LDS threshold to the
; loop's own, so the LDS check skips the GEP and never admits the loop. A loop
; that accesses LDS and is not admitted is never runtime unrolled, even though
; this GEP would pass the check.
; CHECK-LABEL: @lds_check_skipped(
; CHECK-NOT:   for.body.epil:
; CHECK:       ret void
define void @lds_check_skipped(ptr addrspace(1) %out, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %acc = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %idx = zext i32 %iv to i64
  %ptr = getelementptr inbounds [1024 x float], ptr addrspace(3) @lds, i64 0, i64 %idx
  %val = load float, ptr addrspace(3) %ptr, align 4
  %add = fadd float %acc, %val
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !3

exit:
  %res = phi float [ 0.0, %entry ], [ %add, %for.body ]
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}

; No LDS: follows -amdgpu-unroll-runtime-default only.
; CHECK-LABEL: @no_lds(
; DEF-ON:      for.body.epil:
; DEF-OFF-NOT: for.body.epil:
; CHECK:       ret void
define void @no_lds(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %acc = phi float [ 0.0, %entry ], [ %add, %for.body ]
  %ptr = getelementptr inbounds float, ptr addrspace(1) %in, i32 %iv
  %val = load float, ptr addrspace(1) %ptr, align 4
  %add = fadd float %acc, %val
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  %res = phi float [ 0.0, %entry ], [ %add, %for.body ]
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !1}
!3 = distinct !{!3, !4}
!4 = !{!"amdgpu.loop.unroll.threshold", i32 100}
