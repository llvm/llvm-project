; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 -O0 | FileCheck %s --check-prefix PTX
; RUN: opt < %s -S -nvptx-aa -nvptx-aa-wrapper -nvptx-lower-aggr-copies | FileCheck %s --check-prefix IR
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -O0 | %ptxas-verify -arch=sm_90 %}

; Verify that the NVPTXLowerAggrCopies pass works as expected - calls to
; llvm.mem* intrinsics get lowered to loops.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "nvptx64-unknown-unknown"

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #1
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #1
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) #1

define ptr @memcpy_caller(ptr %dst, ptr %src, i64 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret ptr %dst

; IR-LABEL:   @memcpy_caller
; IR:         entry:
; IR:         [[Cond:%[0-9]+]] = icmp ne i64 %n, 0
; IR:         br i1 [[Cond]], label %dynamic-memcpy-expansion-main-body, label %dynamic-memcpy-post-expansion

; IR:         dynamic-memcpy-expansion-main-body:
; IR:         %loop-index = phi i64 [ 0, %entry ], [ [[IndexInc:%[0-9]+]], %dynamic-memcpy-expansion-main-body ]
; IR:         [[SrcGep:%[0-9]+]] = getelementptr inbounds i8, ptr %src, i64 %loop-index
; IR:         [[Load:%[0-9]+]] = load i8, ptr [[SrcGep]]
; IR:         [[DstGep:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64 %loop-index
; IR:         store i8 [[Load]], ptr [[DstGep]]
; IR:         [[IndexInc]] = add i64 %loop-index, 1
; IR:         [[Cond2:%[0-9]+]] = icmp ult i64 [[IndexInc]], %n
; IR:         br i1 [[Cond2]], label %dynamic-memcpy-expansion-main-body, label %dynamic-memcpy-post-expansion

; IR-LABEL:   dynamic-memcpy-post-expansion:
; IR:         ret ptr %dst

; PTX-LABEL:  .visible .func (.param .b64 func_retval0) memcpy_caller
; PTX:        $L__BB[[LABEL:[_0-9]+]]:
; PTX:        ld.b8 %rs[[REG:[0-9]+]]
; PTX:        st.b8 [%rd{{[0-9]+}}], %rs[[REG]]
; PTX:        add.s64 %rd[[COUNTER:[0-9]+]], %rd{{[0-9]+}}, 1
; PTX:        setp.lt.u64 %p[[PRED:[0-9]+]], %rd[[COUNTER]], %rd
; PTX:        @%p[[PRED]] bra $L__BB[[LABEL]]

}

define ptr @memcpy_volatile_caller(ptr %dst, ptr %src, i64 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 true)
  ret ptr %dst

; IR-LABEL:   @memcpy_volatile_caller
; IR:         entry:
; IR:         [[Cond:%[0-9]+]] = icmp ne i64 %n, 0
; IR:         br i1 [[Cond]], label %dynamic-memcpy-expansion-main-body, label %dynamic-memcpy-post-expansion

; IR:         dynamic-memcpy-expansion-main-body:
; IR:         %loop-index = phi i64 [ 0, %entry ], [ [[IndexInc:%[0-9]+]], %dynamic-memcpy-expansion-main-body ]
; IR:         [[SrcGep:%[0-9]+]] = getelementptr inbounds i8, ptr %src, i64 %loop-index
; IR:         [[Load:%[0-9]+]] = load volatile i8, ptr [[SrcGep]]
; IR:         [[DstGep:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64 %loop-index
; IR:         store volatile i8 [[Load]], ptr [[DstGep]]
; IR:         [[IndexInc]] = add i64 %loop-index, 1
; IR:         [[Cond2:%[0-9]+]] = icmp ult i64 [[IndexInc]], %n
; IR:         br i1 [[Cond2]], label %dynamic-memcpy-expansion-main-body, label %dynamic-memcpy-post-expansion

; IR-LABEL:   dynamic-memcpy-post-expansion:
; IR:         ret ptr %dst


; PTX-LABEL:  .visible .func (.param .b64 func_retval0) memcpy_volatile_caller
; PTX:        $L__BB[[LABEL:[_0-9]+]]:
; PTX:        ld.volatile.b8 %rs[[REG:[0-9]+]]
; PTX:        st.volatile.b8 [%rd{{[0-9]+}}], %rs[[REG]]
; PTX:        add.s64 %rd[[COUNTER:[0-9]+]], %rd{{[0-9]+}}, 1
; PTX:        setp.lt.u64 %p[[PRED:[0-9]+]], %rd[[COUNTER]], %rd
; PTX:        @%p[[PRED]] bra $L__BB[[LABEL]]
}

define ptr @memcpy_casting_caller(ptr %dst, ptr %src, i64 %n) #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret ptr %dst

; Check that casts in calls to memcpy are handled properly
; IR-LABEL:   @memcpy_casting_caller
; IR:         getelementptr inbounds i8, ptr %src
; IR:         getelementptr inbounds i8, ptr %dst
}

define ptr @memcpy_known_size(ptr %dst, ptr %src) {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 144, i1 false)
  ret ptr %dst

; Check that calls with compile-time constant size are handled correctly
; IR-LABEL:    @memcpy_known_size
; IR:          entry:
; IR:          br label %static-memcpy-expansion-main-body
; IR:          static-memcpy-expansion-main-body:
; IR:          %loop-index = phi i64 [ 0, %entry ], [ [[IndexInc:%[0-9]+]], %static-memcpy-expansion-main-body ]
; IR:          [[SrcGep:%[0-9]+]] = getelementptr inbounds i8, ptr %src, i64 %loop-index
; IR:          [[Load:%[0-9]+]] = load i8, ptr [[SrcGep]]
; IR:          [[DstGep:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64 %loop-index
; IR:          store i8 [[Load]], ptr [[DstGep]]
; IR:          [[IndexInc]] = add i64 %loop-index, 1
; IR:          [[Cond:%[0-9]+]] = icmp ult i64 %3, 144
; IR:          br i1 [[Cond]], label %static-memcpy-expansion-main-body, label %static-memcpy-post-expansion
}

define ptr @memset_caller(ptr %dst, i32 %c, i64 %n) #0 {
entry:
  %0 = trunc i32 %c to i8
  tail call void @llvm.memset.p0.i64(ptr %dst, i8 %0, i64 %n, i1 false)
  ret ptr %dst

; IR-LABEL:   @memset_caller
; IR:         [[VAL:%[0-9]+]] = trunc i32 %c to i8
; IR:         [[CMPREG:%[0-9]+]] = icmp ne i64 %n, 0
; IR:         br i1 [[CMPREG]], label %dynamic-memset-expansion-main-body, label %dynamic-memset-post-expansion
; IR:         dynamic-memset-expansion-main-body:
; IR:         [[STOREPTR:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64
; IR-NEXT:    store i8 [[VAL]], ptr [[STOREPTR]]

; PTX-LABEL:  .visible .func (.param .b64 func_retval0) memset_caller(
; PTX:        ld.param.b32 %r[[C:[0-9]+]]
; PTX:        cvt.u16.u32  %rs[[REG:[0-9]+]], %r[[C]];
; PTX:        $L__BB[[LABEL:[_0-9]+]]:
; PTX:        st.b8 [%rd{{[0-9]+}}], %rs[[REG]]
; PTX:        add.s64 %rd[[COUNTER:[0-9]+]], %rd{{[0-9]+}}, 1
; PTX:        setp.lt.u64 %p[[PRED:[0-9]+]], %rd[[COUNTER]], %rd
; PTX:        @%p[[PRED]] bra $L__BB[[LABEL]]
}

define ptr @volatile_memset_caller(ptr %dst, i32 %c, i64 %n) #0 {
entry:
  %0 = trunc i32 %c to i8
  tail call void @llvm.memset.p0.i64(ptr %dst, i8 %0, i64 %n, i1 true)
  ret ptr %dst

; IR-LABEL:   @volatile_memset_caller
; IR:         [[VAL:%[0-9]+]] = trunc i32 %c to i8
; IR:         dynamic-memset-expansion-main-body:
; IR:         [[STOREPTR:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64
; IR-NEXT:    store volatile i8 [[VAL]], ptr [[STOREPTR]]
}

define ptr @memmove_caller(ptr %dst, ptr %src, i64 %n) #0 {
entry:
  tail call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  ret ptr %dst

; IR-LABEL:   @memmove_caller
; IR:         icmp ult ptr %src, %dst
; IR:         [[PHIVAL:%[0-9a-zA-Z_]+]] = phi i64
; IR-NEXT:    %bwd_main_index = sub i64 [[PHIVAL]], 1
; IR:         [[FWDPHIVAL:%[0-9a-zA-Z_]+]] = phi i64
; IR:         {{%[0-9a-zA-Z_]+}} = add i64 [[FWDPHIVAL]], 1

; PTX-LABEL:  .visible .func (.param .b64 func_retval0) memmove_caller(
; PTX:        ld.param.b64 %rd[[N:[0-9]+]]
; PTX-DAG:    setp.eq.b64 %p[[NEQ0:[0-9]+]], %rd[[N]], 0
; PTX-DAG:    setp.ge.u64 %p[[SRC_GT_THAN_DST:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; PTX-NEXT:   @%p[[SRC_GT_THAN_DST]] bra $L__BB[[FORWARD_BB:[0-9_]+]]
; -- this is the backwards copying BB
; PTX:        @%p[[NEQ0]] bra $L__BB[[EXIT:[0-9_]+]]
; PTX:        add.s64 %rd{{[0-9]}}, %rd{{[0-9]}}, -1
; PTX:        ld.b8 %rs[[ELEMENT:[0-9]+]]
; PTX:        st.b8 [%rd{{[0-9]+}}], %rs[[ELEMENT]]
; -- this is the forwards copying BB
; PTX:        $L__BB[[FORWARD_BB]]:
; PTX:        @%p[[NEQ0]] bra $L__BB[[EXIT]]
; PTX:        ld.b8 %rs[[ELEMENT2:[0-9]+]]
; PTX:        st.b8 [%rd{{[0-9]+}}], %rs[[ELEMENT2]]
; PTX:        add.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, 1
; -- exit block
; PTX:        $L__BB[[EXIT]]:
; PTX-NEXT:   st.param.b64 [func_retval0
; PTX-NEXT:   ret
}

define void @aggr_loadstore_overlap_forward_copy(ptr %p) {
entry:
  %dst = getelementptr inbounds i8, ptr %p, i64 8
  %v = load [128 x i8], ptr %p, align 1
  store [128 x i8] %v, ptr %dst, align 1
  ret void

; A large aggregate load;store pair may have overlapping src/dst and is fully
; defined (whole value read before any byte stored). It must be lowered to
; overlap-safe memmove-style code (runtime src<dst direction check + backward
; loop), not an unconditional forward copy loop.
; IR-LABEL:   @aggr_loadstore_overlap_forward_copy
; IR:         [[CMP:%[0-9a-zA-Z_]+]] = icmp ult ptr %p, %dst
; IR:         br i1 [[CMP]], label %memmove_bwd_loop, label %memmove_fwd_loop
; IR:         memmove_bwd_loop:
; IR:         %bwd_index = sub i32 {{%[0-9a-zA-Z_]+}}, 1
; IR:         memmove_fwd_loop:
; IR:         {{%[0-9a-zA-Z_]+}} = add i32 %fwd_index, 1

; PTX-LABEL:  .visible .func aggr_loadstore_overlap_forward_copy(
; PTX:        setp.ge.u64 %p{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; PTX:        // %memmove_bwd_loop
; PTX:        // %memmove_fwd_loop
}

define void @aggr_loadstore_generic_global(ptr %g, ptr addrspace(1) %glob) {
  %v = load [128 x i8], ptr %g, align 1
  store [128 x i8] %v, ptr addrspace(1) %glob, align 1
  ret void

; The generic address space aliases every space, so a generic and a global
; pointer may overlap and the copy must be direction-safe. The two pointers
; live in different spaces, so both are cast to generic to make the runtime
; comparison well defined before emitting the memmove-style loop.
; IR-LABEL:   @aggr_loadstore_generic_global
; IR:         [[GG:%[0-9a-zA-Z_]+]] = addrspacecast ptr addrspace(1) %glob to ptr
; IR:         [[CMP:%[0-9a-zA-Z_]+]] = icmp ult ptr %g, [[GG]]
; IR:         br i1 [[CMP]], label %memmove_bwd_loop, label %memmove_fwd_loop
; IR:         memmove_fwd_loop:

; PTX-LABEL:  .visible .func aggr_loadstore_generic_global(
; PTX:        cvta.global.u64
; PTX:        // %memmove_bwd_loop
; PTX:        // %memmove_fwd_loop
}

define void @aggr_loadstore_global_shared(ptr addrspace(1) %glob, ptr addrspace(3) %sh) {
  %v = load [128 x i8], ptr addrspace(1) %glob, align 1
  store [128 x i8] %v, ptr addrspace(3) %sh, align 1
  ret void

; Distinct non-generic address spaces (global vs shared) cannot overlap; this
; fact comes from NVPTXAAResult (the RUN line adds nvptx-aa). So it lowers to a
; plain forward copy loop with no runtime direction check, and the loads/stores
; carry alias-scope/noalias metadata.
; IR-LABEL:   @aggr_loadstore_global_shared
; IR-NOT:     memmove_bwd_loop
; IR-NOT:     addrspacecast
; IR:         load i8, ptr addrspace(1) {{.*}}, !alias.scope
; IR:         store i8 {{.*}}, ptr addrspace(3) {{.*}}, !noalias

; PTX-LABEL:  .visible .func aggr_loadstore_global_shared(
; PTX:        // %static-memcpy
; PTX-NOT:    // %memmove_bwd_loop
}

define void @aggr_loadstore_shared_cluster(ptr addrspace(3) %sh, ptr addrspace(7) %clus) {
  %v = load [128 x i8], ptr addrspace(3) %sh, align 1
  store [128 x i8] %v, ptr addrspace(7) %clus, align 1
  ret void

; Distributed shared (addrspace 7) aliases shared (addrspace 3), so this pair
; may overlap. The spaces differ, so both pointers are cast to generic for the
; runtime comparison; cvta.shared / cvta.shared::cluster make that legal.
; IR-LABEL:   @aggr_loadstore_shared_cluster
; IR-DAG:     [[S:%[0-9a-zA-Z_]+]] = addrspacecast ptr addrspace(3) %sh to ptr
; IR-DAG:     [[C:%[0-9a-zA-Z_]+]] = addrspacecast ptr addrspace(7) %clus to ptr
; IR:         [[CMP:%[0-9a-zA-Z_]+]] = icmp ult ptr [[S]], [[C]]
; IR:         br i1 [[CMP]], label %memmove_bwd_loop, label %memmove_fwd_loop

; PTX-LABEL:  .visible .func aggr_loadstore_shared_cluster(
; PTX:        cvta.shared.u64
; PTX:        cvta.shared::cluster.u64
; PTX:        // %memmove_bwd_loop
; PTX:        // %memmove_fwd_loop
}

define void @aggr_loadstore_same_global(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %v = load [128 x i8], ptr addrspace(1) %a, align 1
  store [128 x i8] %v, ptr addrspace(1) %b, align 1
  ret void

; Same (non-generic) address space: may overlap, but the pointers are already
; comparable, so no addrspacecast is introduced and the overlap-safe loop runs
; directly in that space.
; IR-LABEL:   @aggr_loadstore_same_global
; IR-NOT:     addrspacecast
; IR:         icmp ult ptr addrspace(1) %a, %b
; IR:         load i8, ptr addrspace(1)
; IR:         store i8 {{.*}}, ptr addrspace(1)
}

define void @aggr_loadstore_noalias(ptr noalias %dst, ptr noalias %src) {
  %v = load [128 x i8], ptr %src, align 1
  store [128 x i8] %v, ptr %dst, align 1
  ret void

; noalias pointers can't overlap and BasicAA proves it (so this holds under opt
; too). The copy is a plain forward loop carrying alias-scope/noalias metadata,
; with no runtime direction check and no addrspacecast.
; IR-LABEL:   @aggr_loadstore_noalias
; IR-NOT:     memmove_bwd_loop
; IR-NOT:     addrspacecast
; IR:         load i8, ptr {{.*}}, !alias.scope
; IR:         store i8 {{.*}}, !noalias
}
