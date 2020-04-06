; RUN: opt -load-store-vectorizer -S < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p5:32:32"

; Size mismatch between the 32 bit pointer in address space 5 and 64 bit
; pointer in address space 0 it was cast to caused below test to crash.
; The p5:32:32 portion of the data layout is critical for the test.

; CHECK-LABEL: @cast_to_ptr
; CHECK: store i32* undef, i32** %tmp9, align 8
; CHECK: store i32* undef, i32** %tmp7, align 8
define void @cast_to_ptr() {
entry:
  %ascast = addrspacecast i32* addrspace(5)* null to i32**
  %tmp4 = icmp eq i32 undef, 0
  %tmp6 = select i1 false, i32** undef, i32** undef
  %tmp7 = select i1 %tmp4, i32** null, i32** %tmp6
  %tmp9 = select i1 %tmp4, i32** %ascast, i32** null
  store i32* undef, i32** %tmp9, align 8
  store i32* undef, i32** %tmp7, align 8
  unreachable
}

; CHECK-LABEL: @cast_to_cast
; CHECK: %tmp4 = load i32*, i32** %tmp1, align 8
; CHECK: %tmp5 = load i32*, i32** %tmp3, align 8
define void @cast_to_cast() {
entry:
  %a.ascast = addrspacecast i32* addrspace(5)* undef to i32**
  %b.ascast = addrspacecast i32* addrspace(5)* null to i32**
  %tmp1 = select i1 false, i32** %a.ascast, i32** undef
  %tmp3 = select i1 false, i32** %b.ascast, i32** undef
  %tmp4 = load i32*, i32** %tmp1, align 8
  %tmp5 = load i32*, i32** %tmp3, align 8
  unreachable
}

; CHECK-LABEL: @all_to_cast
; CHECK: load <4 x float>
define void @all_to_cast(i8* nocapture readonly align 16 dereferenceable(16) %alloc1) {
entry:
  %alloc16 = addrspacecast i8* %alloc1 to i8 addrspace(1)*
  %tmp = bitcast i8 addrspace(1)* %alloc16 to float addrspace(1)*
  %tmp1 = load float, float addrspace(1)* %tmp, align 16, !invariant.load !0
  %tmp6 = getelementptr inbounds i8, i8 addrspace(1)* %alloc16, i64 4
  %tmp7 = bitcast i8 addrspace(1)* %tmp6 to float addrspace(1)*
  %tmp8 = load float, float addrspace(1)* %tmp7, align 4, !invariant.load !0
  %tmp15 = getelementptr inbounds i8, i8 addrspace(1)* %alloc16, i64 8
  %tmp16 = bitcast i8 addrspace(1)* %tmp15 to float addrspace(1)*
  %tmp17 = load float, float addrspace(1)* %tmp16, align 8, !invariant.load !0
  %tmp24 = getelementptr inbounds i8, i8 addrspace(1)* %alloc16, i64 12
  %tmp25 = bitcast i8 addrspace(1)* %tmp24 to float addrspace(1)*
  %tmp26 = load float, float addrspace(1)* %tmp25, align 4, !invariant.load !0
  ret void
}

; CHECK-LABEL: @ext_ptr
; CHECK: load <2 x i32>
define void @ext_ptr(i32 addrspace(5)* %p) {
entry:
  %gep1 = getelementptr inbounds i32, i32 addrspace(5)* %p, i64 0
  %gep2 = getelementptr inbounds i32, i32 addrspace(5)* %p, i64 1
  %a.ascast = addrspacecast i32 addrspace(5)* %gep1 to i32*
  %b.ascast = addrspacecast i32 addrspace(5)* %gep2 to i32*
  %tmp1 = load i32, i32* %a.ascast, align 8
  %tmp2 = load i32, i32* %b.ascast, align 8
  unreachable
}

; CHECK-LABEL: @shrink_ptr
; CHECK: load <2 x i32>
define void @shrink_ptr(i32* %p) {
entry:
  %gep1 = getelementptr inbounds i32, i32* %p, i64 0
  %gep2 = getelementptr inbounds i32, i32* %p, i64 1
  %a.ascast = addrspacecast i32* %gep1 to i32 addrspace(5)*
  %b.ascast = addrspacecast i32* %gep2 to i32 addrspace(5)*
  %tmp1 = load i32, i32 addrspace(5)* %a.ascast, align 8
  %tmp2 = load i32, i32 addrspace(5)* %b.ascast, align 8
  unreachable
}

; CHECK-LABEL: @ext_ptr_wrap
; CHECK: load <2 x i8>
define void @ext_ptr_wrap(i8 addrspace(5)* %p) {
entry:
  %gep1 = getelementptr inbounds i8, i8 addrspace(5)* %p, i64 0
  %gep2 = getelementptr inbounds i8, i8 addrspace(5)* %p, i64 4294967295
  %a.ascast = addrspacecast i8 addrspace(5)* %gep1 to i8*
  %b.ascast = addrspacecast i8 addrspace(5)* %gep2 to i8*
  %tmp1 = load i8, i8* %a.ascast, align 1
  %tmp2 = load i8, i8* %b.ascast, align 1
  unreachable
}

!0 = !{}
