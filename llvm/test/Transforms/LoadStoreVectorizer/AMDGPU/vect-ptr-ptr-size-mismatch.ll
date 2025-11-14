; RUN: opt -passes=load-store-vectorizer -S < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p5:32:32"

; Size mismatch between the 32 bit pointer in address space 5 and 64 bit
; pointer in address space 0 it was cast to caused below test to crash.
; The p5:32:32 portion of the data layout is critical for the test.

; CHECK-LABEL: @cast_to_ptr
; CHECK: store ptr undef, ptr %tmp9, align 8
; CHECK: store ptr undef, ptr %tmp7, align 8
define void @cast_to_ptr() {
entry:
  %ascast = addrspacecast ptr addrspace(5) null to ptr
  %tmp4 = icmp eq i32 undef, 0
  %tmp6 = select i1 false, ptr undef, ptr undef
  %tmp7 = select i1 %tmp4, ptr null, ptr %tmp6
  %tmp9 = select i1 %tmp4, ptr %ascast, ptr null
  store ptr undef, ptr %tmp9, align 8
  store ptr undef, ptr %tmp7, align 8
  unreachable
}

; CHECK-LABEL: @cast_to_cast
; CHECK: %tmp4 = load ptr, ptr %tmp1, align 8
; CHECK: %tmp5 = load ptr, ptr %tmp3, align 8
define void @cast_to_cast() {
entry:
  %a.ascast = addrspacecast ptr addrspace(5) undef to ptr
  %b.ascast = addrspacecast ptr addrspace(5) null to ptr
  %tmp1 = select i1 false, ptr %a.ascast, ptr undef
  %tmp3 = select i1 false, ptr %b.ascast, ptr undef
  %tmp4 = load ptr, ptr %tmp1, align 8
  %tmp5 = load ptr, ptr %tmp3, align 8
  unreachable
}

; CHECK-LABEL: @all_to_cast
; CHECK: load <4 x float>
define void @all_to_cast(ptr nocapture readonly align 16 dereferenceable(16) %alloc1) {
entry:
  %alloc16 = addrspacecast ptr %alloc1 to ptr addrspace(1)
  %tmp1 = load float, ptr addrspace(1) %alloc16, align 16, !invariant.load !0
  %tmp6 = getelementptr inbounds i8, ptr addrspace(1) %alloc16, i64 4
  %tmp8 = load float, ptr addrspace(1) %tmp6, align 4, !invariant.load !0
  %tmp15 = getelementptr inbounds i8, ptr addrspace(1) %alloc16, i64 8
  %tmp17 = load float, ptr addrspace(1) %tmp15, align 8, !invariant.load !0
  %tmp24 = getelementptr inbounds i8, ptr addrspace(1) %alloc16, i64 12
  %tmp26 = load float, ptr addrspace(1) %tmp24, align 4, !invariant.load !0
  ret void
}

; CHECK-LABEL: @ext_ptr
; CHECK: load <2 x i32>
define void @ext_ptr(ptr addrspace(5) %p) {
entry:
  %gep2 = getelementptr inbounds i32, ptr addrspace(5) %p, i64 1
  %a.ascast = addrspacecast ptr addrspace(5) %p to ptr
  %b.ascast = addrspacecast ptr addrspace(5) %gep2 to ptr
  %tmp1 = load i32, ptr %a.ascast, align 8
  %tmp2 = load i32, ptr %b.ascast, align 8
  unreachable
}

; CHECK-LABEL: @select_different_as
; CHECK: load <2 x i32>
define void @select_different_as(ptr addrspace(1) %p0, ptr addrspace(5) %q0, i1 %cond) {
entry:
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 1
  %q1 = getelementptr inbounds i32, ptr addrspace(5) %q0, i64 1
  %p0.ascast = addrspacecast ptr addrspace(1) %p0 to ptr
  %p1.ascast = addrspacecast ptr addrspace(1) %p1 to ptr
  %q0.ascast = addrspacecast ptr addrspace(5) %q0 to ptr
  %q1.ascast = addrspacecast ptr addrspace(5) %q1 to ptr
  %sel0 = select i1 %cond, ptr %p0.ascast, ptr %q0.ascast
  %sel1 = select i1 %cond, ptr %p1.ascast, ptr %q1.ascast
  %tmp1 = load i32, ptr %sel0, align 8
  %tmp2 = load i32, ptr %sel1, align 8
  unreachable
}

; CHECK-LABEL: @shrink_ptr
; CHECK: load <2 x i32>
define void @shrink_ptr(ptr %p) {
entry:
  %gep2 = getelementptr inbounds i32, ptr %p, i64 1
  %a.ascast = addrspacecast ptr %p to ptr addrspace(5)
  %b.ascast = addrspacecast ptr %gep2 to ptr addrspace(5)
  %tmp1 = load i32, ptr addrspace(5) %a.ascast, align 8
  %tmp2 = load i32, ptr addrspace(5) %b.ascast, align 8
  unreachable
}

; CHECK-LABEL: @ext_ptr_wrap
; CHECK: load <2 x i8>
define void @ext_ptr_wrap(ptr addrspace(5) %p) {
entry:
  %gep2 = getelementptr inbounds i8, ptr addrspace(5) %p, i64 4294967295
  %a.ascast = addrspacecast ptr addrspace(5) %p to ptr
  %b.ascast = addrspacecast ptr addrspace(5) %gep2 to ptr
  %tmp1 = load i8, ptr %a.ascast, align 1
  %tmp2 = load i8, ptr %b.ascast, align 2
  unreachable
}

!0 = !{}
