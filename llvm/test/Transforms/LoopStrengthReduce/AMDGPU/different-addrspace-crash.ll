; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--"

; We need to compile this for a target where we have different address spaces,
; and where pointers in those address spaces have different size.
; E.g. for amdgcn-- pointers in address space 0 are 32 bits and pointers in
; address space 1 are 64 bits.

; We shouldn't crash. Check that we get a loop with the two stores.
;CHECK-LABEL: foo:
;CHECK: [[LOOP_LABEL:BB[0-9]+_[0-9]+]]:
;CHECK: buffer_store_dword
;CHECK: buffer_store_dword
;CHECK: s_branch [[LOOP_LABEL]]

define void @foo() {
entry:
  br label %loop

loop:
  %idx0 = phi i32 [ %next_idx0, %loop ], [ 0, %entry ]
  %0 = getelementptr inbounds i32, i32* null, i32 %idx0
  %1 = getelementptr inbounds i32, i32 addrspace(1)* null, i32 %idx0
  store i32 1, i32* %0
  store i32 7, i32 addrspace(1)* %1
  %next_idx0 = add nuw nsw i32 %idx0, 1
  br label %loop
}

