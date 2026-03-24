; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-promote-alloca -amdgpu-promote-alloca-to-vector-limit=512 %s | FileCheck %s

; Regression test for stale state with a dynamic promoted index.
; Earlier alloca promotion can rewrite the value used as a later alloca's
; dynamic vector index. If the later alloca reuses stale cached analysis state
; and falls back to index 0, this becomes obvious wrong code: the transformed IR
; would read and write b[0] instead of b[idx].

define amdgpu_kernel void @dynamic_index_stale_state(ptr addrspace(1) %out, i32 %sel) #0 {
; CHECK-LABEL: define amdgpu_kernel void @dynamic_index_stale_state(
; CHECK-SAME: ptr addrspace(1) [[OUT:%.*]], i32 [[SEL:%.*]]) #[[ATTRS:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = freeze <4 x i32> poison
; CHECK-NEXT:    %a = freeze <4 x i32> poison
; CHECK-NEXT:    [[IDX:%.*]] = and i32 [[SEL]], 3
; CHECK:         [[B0:%.*]] = insertelement <4 x i32> {{.*}}, i32 11, i32 0
; CHECK-NEXT:    [[B1:%.*]] = insertelement <4 x i32> [[B0]], i32 22, i32 1
; CHECK-NEXT:    [[B2:%.*]] = insertelement <4 x i32> [[B1]], i32 33, i32 2
; CHECK-NEXT:    [[B3:%.*]] = insertelement <4 x i32> [[B2]], i32 44, i32 3
; CHECK:         [[OLD:%.*]] = extractelement <4 x i32> [[B3]], i32 [[IDX]]
; CHECK-NEXT:    [[NEW:%.*]] = add i32 [[OLD]], 100
; CHECK-NEXT:    [[B4:%.*]] = insertelement <4 x i32> [[B3]], i32 [[NEW]], i32 [[IDX]]
; CHECK-NEXT:    store i32 [[NEW]], ptr addrspace(1) [[OUT]], align 4
; CHECK-NEXT:    ret void
entry:
  %a = alloca [4 x i32], align 4, addrspace(5)
  %b = alloca [4 x i32], align 4, addrspace(5)

  ; Keep the dynamic index in range but still data-dependent.
  %idx = and i32 %sel, 3

  ; Give %a enough users that it stays first in the sorted promotion worklist.
  %a.ptr0 = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 0
  store i32 %idx, ptr addrspace(5) %a.ptr0, align 4
  %a.idx0 = load i32, ptr addrspace(5) %a.ptr0, align 4
  %a.ptr1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 1
  store i32 %a.idx0, ptr addrspace(5) %a.ptr1, align 4
  %a.idx1 = load i32, ptr addrspace(5) %a.ptr1, align 4
  %a.ptr2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 2
  store i32 %a.idx1, ptr addrspace(5) %a.ptr2, align 4
  %a.idx2 = load i32, ptr addrspace(5) %a.ptr2, align 4

  ; Initialize %b with distinct values so collapsing the dynamic index to 0 is
  ; clearly visible in the promoted IR.
  %b.ptr0 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 0
  store i32 11, ptr addrspace(5) %b.ptr0, align 4
  %b.ptr1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 1
  store i32 22, ptr addrspace(5) %b.ptr1, align 4
  %b.ptr2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 2
  store i32 33, ptr addrspace(5) %b.ptr2, align 4
  %b.ptr3 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 3
  store i32 44, ptr addrspace(5) %b.ptr3, align 4

  ; %b's cached vector index is based on the load chain from %a. After %a is
  ; promoted, the replacement value must still be used here rather than
  ; incorrectly falling back to element 0.
  %b.ptr.dynamic = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 %a.idx2
  %old = load i32, ptr addrspace(5) %b.ptr.dynamic, align 4
  %new = add i32 %old, 100
  store i32 %new, ptr addrspace(5) %b.ptr.dynamic, align 4
  store i32 %new, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "amdgpu-promote-alloca-to-vector-max-regs"="1024" }
