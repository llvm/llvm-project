; RUN: llc -O0 -verify-machineinstrs -asm-verbose=0 -mtriple=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=COMMON %s
; RUN: llc -O0 -verify-machineinstrs -asm-verbose=0 -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=COMMON %s
; XUN: llc -O0 -verify-machineinstrs -asm-verbose=0 -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=COMMON %s

; SI hits an assertion at -O0, evergreen hits a not implemented unreachable.

; COMMON-LABEL: {{^}}branch_true:
define amdgpu_kernel void @branch_true(ptr addrspace(1) nocapture %main, i32 %main_stride) #0 {
entry:
  br i1 true, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr.sum = shl i32 %main_stride, 1
  %add.ptr1.sum = add i32 %add.ptr.sum, %main_stride
  %add.ptr4.sum = shl i32 %main_stride, 2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %main.addr.011 = phi ptr addrspace(1) [ %main, %for.body.lr.ph ], [ %add.ptr6, %for.body ]
  %0 = load i32, ptr addrspace(1) %main.addr.011, align 4
  %add.ptr = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %main_stride
  %1 = load i32, ptr addrspace(1) %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr.sum
  %2 = load i32, ptr addrspace(1) %add.ptr1, align 4
  %add.ptr2 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr1.sum
  %3 = load i32, ptr addrspace(1) %add.ptr2, align 4
  %add.ptr3 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr4.sum
  %4 = load i32, ptr addrspace(1) %add.ptr3, align 4
  %add.ptr6 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 undef
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; COMMON-LABEL: {{^}}branch_false:
; SI: s_cbranch_scc1
; SI: s_endpgm
define amdgpu_kernel void @branch_false(ptr addrspace(1) nocapture %main, i32 %main_stride) #0 {
entry:
  br i1 false, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr.sum = shl i32 %main_stride, 1
  %add.ptr1.sum = add i32 %add.ptr.sum, %main_stride
  %add.ptr4.sum = shl i32 %main_stride, 2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %main.addr.011 = phi ptr addrspace(1) [ %main, %for.body.lr.ph ], [ %add.ptr6, %for.body ]
  %0 = load i32, ptr addrspace(1) %main.addr.011, align 4
  %add.ptr = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %main_stride
  %1 = load i32, ptr addrspace(1) %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr.sum
  %2 = load i32, ptr addrspace(1) %add.ptr1, align 4
  %add.ptr2 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr1.sum
  %3 = load i32, ptr addrspace(1) %add.ptr2, align 4
  %add.ptr3 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr4.sum
  %4 = load i32, ptr addrspace(1) %add.ptr3, align 4
  %add.ptr6 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 undef
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; COMMON-LABEL: {{^}}branch_undef:
; SI: s_cbranch_scc1
; SI: s_cbranch_scc1
; SI: s_endpgm
define amdgpu_kernel void @branch_undef(ptr addrspace(1) nocapture %main, i32 %main_stride) #0 {
entry:
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr.sum = shl i32 %main_stride, 1
  %add.ptr1.sum = add i32 %add.ptr.sum, %main_stride
  %add.ptr4.sum = shl i32 %main_stride, 2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %main.addr.011 = phi ptr addrspace(1) [ %main, %for.body.lr.ph ], [ %add.ptr6, %for.body ]
  %0 = load i32, ptr addrspace(1) %main.addr.011, align 4
  %add.ptr = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %main_stride
  %1 = load i32, ptr addrspace(1) %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr.sum
  %2 = load i32, ptr addrspace(1) %add.ptr1, align 4
  %add.ptr2 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr1.sum
  %3 = load i32, ptr addrspace(1) %add.ptr2, align 4
  %add.ptr3 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 %add.ptr4.sum
  %4 = load i32, ptr addrspace(1) %add.ptr3, align 4
  %add.ptr6 = getelementptr inbounds i8, ptr addrspace(1) %main.addr.011, i32 undef
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind }
