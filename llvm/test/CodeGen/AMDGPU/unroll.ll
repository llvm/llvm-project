; RUN: opt -mtriple=amdgcn-- -passes='loop-unroll,simplifycfg,sroa' %s -S -o - | FileCheck %s
; RUN: opt -mtriple=r600-- -passes='loop-unroll,simplifycfg,sroa' %s -S -o - | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; This test contains a simple loop that initializes an array declared in
; private memory.  We want to make sure these kinds of loops are always
; unrolled, because private memory is slow.

; CHECK-LABEL: @private_memory
; CHECK-NOT: alloca
; CHECK: store i32 5, ptr addrspace(1) %out
define amdgpu_kernel void @private_memory(ptr addrspace(1) %out) {
entry:
  %0 = alloca [32 x i32], addrspace(5)
  br label %loop.header

loop.header:
  %counter = phi i32 [0, %entry], [%inc, %loop.inc]
  br label %loop.body

loop.body:
  %ptr = getelementptr [32 x i32], ptr addrspace(5) %0, i32 0, i32 %counter
  store i32 %counter, ptr addrspace(5) %ptr
  br label %loop.inc

loop.inc:
  %inc = add i32 %counter, 1
  %1 = icmp sge i32 %counter, 32
  br i1 %1, label  %exit, label %loop.header

exit:
  %2 = getelementptr [32 x i32], ptr addrspace(5) %0, i32 0, i32 5
  %3 = load i32, ptr addrspace(5) %2
  store i32 %3, ptr addrspace(1) %out
  ret void
}

; Check that loop is unrolled for local memory references

; CHECK-LABEL: @local_memory
; CHECK: getelementptr i32, ptr addrspace(1) %out, i32 128
; CHECK-NEXT: store
; CHECK-NEXT: ret
define amdgpu_kernel void @local_memory(ptr addrspace(1) %out, ptr addrspace(3) %lds) {
entry:
  br label %loop.header

loop.header:
  %counter = phi i32 [0, %entry], [%inc, %loop.inc]
  br label %loop.body

loop.body:
  %ptr_lds = getelementptr i32, ptr addrspace(3) %lds, i32 %counter
  %val = load i32, ptr addrspace(3) %ptr_lds
  %ptr_out = getelementptr i32, ptr addrspace(1) %out, i32 %counter
  store i32 %val, ptr addrspace(1) %ptr_out
  br label %loop.inc

loop.inc:
  %inc = add i32 %counter, 1
  %cond = icmp sge i32 %counter, 128
  br i1 %cond, label  %exit, label %loop.header

exit:
  ret void
}

; Check that a loop with if inside completely unrolled to eliminate phi and if

; CHECK-LABEL: @unroll_for_if
; CHECK: entry:
; CHECK-NEXT: getelementptr
; CHECK-NEXT: store
; CHECK-NEXT: getelementptr
; CHECK-NEXT: store
; CHECK-NOT: br
define amdgpu_kernel void @unroll_for_if(ptr addrspace(5) %a) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i1 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %tobool = icmp eq i32 %i1, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %0 = sext i32 %i1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(5) %a, i64 %0
  store i32 0, ptr addrspace(5) %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i32 %i1, 1
  %cmp = icmp ult i32 %inc, 38
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.cond
  ret void
}

; Check that runtime unroll is enabled for local memory references

; CHECK-LABEL: @local_memory_runtime
; CHECK: loop.header:
; CHECK: load i32, ptr addrspace(3)
; CHECK: load i32, ptr addrspace(3)
; CHECK: br i1
; CHECK: loop.header.epil
; CHECK: load i32, ptr addrspace(3)
; CHECK: ret
define amdgpu_kernel void @local_memory_runtime(ptr addrspace(1) %out, ptr addrspace(3) %lds, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %counter = phi i32 [0, %entry], [%inc, %loop.inc]
  br label %loop.body

loop.body:
  %ptr_lds = getelementptr i32, ptr addrspace(3) %lds, i32 %counter
  %val = load i32, ptr addrspace(3) %ptr_lds
  %ptr_out = getelementptr i32, ptr addrspace(1) %out, i32 %counter
  store i32 %val, ptr addrspace(1) %ptr_out
  br label %loop.inc

loop.inc:
  %inc = add i32 %counter, 1
  %cond = icmp sge i32 %counter, %n
  br i1 %cond, label  %exit, label %loop.header

exit:
  ret void
}
