; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 -O0 | FileCheck %s --check-prefix PTX
; RUN: opt < %s -S -nvptx-lower-aggr-copies | FileCheck %s --check-prefix IR
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 -O0 | %ptxas-verify %}

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
; IR:         br i1 [[Cond]], label %loop-memcpy-expansion, label %post-loop-memcpy-expansion

; IR:         loop-memcpy-expansion:
; IR:         %loop-index = phi i64 [ 0, %entry ], [ [[IndexInc:%[0-9]+]], %loop-memcpy-expansion ]
; IR:         [[SrcGep:%[0-9]+]] = getelementptr inbounds i8, ptr %src, i64 %loop-index
; IR:         [[Load:%[0-9]+]] = load i8, ptr [[SrcGep]]
; IR:         [[DstGep:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64 %loop-index
; IR:         store i8 [[Load]], ptr [[DstGep]]
; IR:         [[IndexInc]] = add i64 %loop-index, 1
; IR:         [[Cond2:%[0-9]+]] = icmp ult i64 [[IndexInc]], %n
; IR:         br i1 [[Cond2]], label %loop-memcpy-expansion, label %post-loop-memcpy-expansion

; IR-LABEL:   post-loop-memcpy-expansion:
; IR:         ret ptr %dst

; PTX-LABEL:  .visible .func (.param .b64 func_retval0) memcpy_caller
; PTX:        $L__BB[[LABEL:[_0-9]+]]:
; PTX:        ld.u8 %rs[[REG:[0-9]+]]
; PTX:        st.u8 [%rd{{[0-9]+}}], %rs[[REG]]
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
; IR:         br i1 [[Cond]], label %loop-memcpy-expansion, label %post-loop-memcpy-expansion

; IR:         loop-memcpy-expansion:
; IR:         %loop-index = phi i64 [ 0, %entry ], [ [[IndexInc:%[0-9]+]], %loop-memcpy-expansion ]
; IR:         [[SrcGep:%[0-9]+]] = getelementptr inbounds i8, ptr %src, i64 %loop-index
; IR:         [[Load:%[0-9]+]] = load volatile i8, ptr [[SrcGep]]
; IR:         [[DstGep:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64 %loop-index
; IR:         store volatile i8 [[Load]], ptr [[DstGep]]
; IR:         [[IndexInc]] = add i64 %loop-index, 1
; IR:         [[Cond2:%[0-9]+]] = icmp ult i64 [[IndexInc]], %n
; IR:         br i1 [[Cond2]], label %loop-memcpy-expansion, label %post-loop-memcpy-expansion

; IR-LABEL:   post-loop-memcpy-expansion:
; IR:         ret ptr %dst


; PTX-LABEL:  .visible .func (.param .b64 func_retval0) memcpy_volatile_caller
; PTX:        $L__BB[[LABEL:[_0-9]+]]:
; PTX:        ld.volatile.u8 %rs[[REG:[0-9]+]]
; PTX:        st.volatile.u8 [%rd{{[0-9]+}}], %rs[[REG]]
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
; IR:          br label %load-store-loop
; IR:          load-store-loop:
; IR:          %loop-index = phi i64 [ 0, %entry ], [ [[IndexInc:%[0-9]+]], %load-store-loop ]
; IR:          [[SrcGep:%[0-9]+]] = getelementptr inbounds i8, ptr %src, i64 %loop-index
; IR:          [[Load:%[0-9]+]] = load i8, ptr [[SrcGep]]
; IR:          [[DstGep:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64 %loop-index
; IR:          store i8 [[Load]], ptr [[DstGep]]
; IR:          [[IndexInc]] = add i64 %loop-index, 1
; IR:          [[Cond:%[0-9]+]] = icmp ult i64 %3, 144
; IR:          br i1 [[Cond]], label %load-store-loop, label %memcpy-split
}

define ptr @memset_caller(ptr %dst, i32 %c, i64 %n) #0 {
entry:
  %0 = trunc i32 %c to i8
  tail call void @llvm.memset.p0.i64(ptr %dst, i8 %0, i64 %n, i1 false)
  ret ptr %dst

; IR-LABEL:   @memset_caller
; IR:         [[VAL:%[0-9]+]] = trunc i32 %c to i8
; IR:         [[CMPREG:%[0-9]+]] = icmp eq i64 0, %n
; IR:         br i1 [[CMPREG]], label %split, label %loadstoreloop
; IR:         loadstoreloop:
; IR:         [[STOREPTR:%[0-9]+]] = getelementptr inbounds i8, ptr %dst, i64
; IR-NEXT:    store i8 [[VAL]], ptr [[STOREPTR]]

; PTX-LABEL:  .visible .func (.param .b64 func_retval0) memset_caller(
; PTX:        ld.param.u32 %r[[C:[0-9]+]]
; PTX:        cvt.u16.u32  %rs[[REG:[0-9]+]], %r[[C]];
; PTX:        $L__BB[[LABEL:[_0-9]+]]:
; PTX:        st.u8 [%rd{{[0-9]+}}], %rs[[REG]]
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
; IR:         loadstoreloop:
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
; PTX:        ld.param.u64 %rd[[N:[0-9]+]]
; PTX-DAG:    setp.eq.s64 %p[[NEQ0:[0-9]+]], %rd[[N]], 0
; PTX-DAG:    setp.ge.u64 %p[[SRC_GT_THAN_DST:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; PTX-NEXT:   @%p[[SRC_GT_THAN_DST]] bra $L__BB[[FORWARD_BB:[0-9_]+]]
; -- this is the backwards copying BB
; PTX:        @%p[[NEQ0]] bra $L__BB[[EXIT:[0-9_]+]]
; PTX:        add.s64 %rd{{[0-9]}}, %rd{{[0-9]}}, -1
; PTX:        ld.u8 %rs[[ELEMENT:[0-9]+]]
; PTX:        st.u8 [%rd{{[0-9]+}}], %rs[[ELEMENT]]
; -- this is the forwards copying BB
; PTX:        $L__BB[[FORWARD_BB]]:
; PTX:        @%p[[NEQ0]] bra $L__BB[[EXIT]]
; PTX:        ld.u8 %rs[[ELEMENT2:[0-9]+]]
; PTX:        st.u8 [%rd{{[0-9]+}}], %rs[[ELEMENT2]]
; PTX:        add.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, 1
; -- exit block
; PTX:        $L__BB[[EXIT]]:
; PTX-NEXT:   st.param.b64 [func_retval0
; PTX-NEXT:   ret
}
