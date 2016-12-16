; RUN: llc -O0 -verify-machineinstrs -asm-verbose=0 -march=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=COMMON %s
; RUN: llc -O0 -verify-machineinstrs -asm-verbose=0 -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=COMMON %s
; XUN: llc -O0 -verify-machineinstrs -asm-verbose=0 -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=COMMON %s

; SI hits an assertion at -O0, evergreen hits a not implemented unreachable.

; COMMON-LABEL: {{^}}branch_true:
define void @branch_true(i8 addrspace(1)* nocapture %main, i32 %main_stride) #0 {
entry:
  br i1 true, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr.sum = shl i32 %main_stride, 1
  %add.ptr1.sum = add i32 %add.ptr.sum, %main_stride
  %add.ptr4.sum = shl i32 %main_stride, 2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %main.addr.011 = phi i8 addrspace(1)* [ %main, %for.body.lr.ph ], [ %add.ptr6, %for.body ]
  %0 = bitcast i8 addrspace(1)* %main.addr.011 to i32 addrspace(1)*
  %1 = load i32, i32 addrspace(1)* %0, align 4
  %add.ptr = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %main_stride
  %2 = bitcast i8 addrspace(1)* %add.ptr to i32 addrspace(1)*
  %3 = load i32, i32 addrspace(1)* %2, align 4
  %add.ptr1 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr.sum
  %4 = bitcast i8 addrspace(1)* %add.ptr1 to i32 addrspace(1)*
  %5 = load i32, i32 addrspace(1)* %4, align 4
  %add.ptr2 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr1.sum
  %6 = bitcast i8 addrspace(1)* %add.ptr2 to i32 addrspace(1)*
  %7 = load i32, i32 addrspace(1)* %6, align 4
  %add.ptr3 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr4.sum
  %8 = bitcast i8 addrspace(1)* %add.ptr3 to i32 addrspace(1)*
  %9 = load i32, i32 addrspace(1)* %8, align 4
  %add.ptr6 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 undef
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; COMMON-LABEL: {{^}}branch_false:
; SI: s_cbranch_vccnz
; SI: s_cbranch_scc1
; SI: s_endpgm
define void @branch_false(i8 addrspace(1)* nocapture %main, i32 %main_stride) #0 {
entry:
  br i1 false, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr.sum = shl i32 %main_stride, 1
  %add.ptr1.sum = add i32 %add.ptr.sum, %main_stride
  %add.ptr4.sum = shl i32 %main_stride, 2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %main.addr.011 = phi i8 addrspace(1)* [ %main, %for.body.lr.ph ], [ %add.ptr6, %for.body ]
  %0 = bitcast i8 addrspace(1)* %main.addr.011 to i32 addrspace(1)*
  %1 = load i32, i32 addrspace(1)* %0, align 4
  %add.ptr = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %main_stride
  %2 = bitcast i8 addrspace(1)* %add.ptr to i32 addrspace(1)*
  %3 = load i32, i32 addrspace(1)* %2, align 4
  %add.ptr1 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr.sum
  %4 = bitcast i8 addrspace(1)* %add.ptr1 to i32 addrspace(1)*
  %5 = load i32, i32 addrspace(1)* %4, align 4
  %add.ptr2 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr1.sum
  %6 = bitcast i8 addrspace(1)* %add.ptr2 to i32 addrspace(1)*
  %7 = load i32, i32 addrspace(1)* %6, align 4
  %add.ptr3 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr4.sum
  %8 = bitcast i8 addrspace(1)* %add.ptr3 to i32 addrspace(1)*
  %9 = load i32, i32 addrspace(1)* %8, align 4
  %add.ptr6 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 undef
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; COMMON-LABEL: {{^}}branch_undef:
; SI: s_cbranch_scc1
; SI: s_cbranch_scc1
; SI: s_endpgm
define void @branch_undef(i8 addrspace(1)* nocapture %main, i32 %main_stride) #0 {
entry:
  br i1 undef, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr.sum = shl i32 %main_stride, 1
  %add.ptr1.sum = add i32 %add.ptr.sum, %main_stride
  %add.ptr4.sum = shl i32 %main_stride, 2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %main.addr.011 = phi i8 addrspace(1)* [ %main, %for.body.lr.ph ], [ %add.ptr6, %for.body ]
  %0 = bitcast i8 addrspace(1)* %main.addr.011 to i32 addrspace(1)*
  %1 = load i32, i32 addrspace(1)* %0, align 4
  %add.ptr = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %main_stride
  %2 = bitcast i8 addrspace(1)* %add.ptr to i32 addrspace(1)*
  %3 = load i32, i32 addrspace(1)* %2, align 4
  %add.ptr1 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr.sum
  %4 = bitcast i8 addrspace(1)* %add.ptr1 to i32 addrspace(1)*
  %5 = load i32, i32 addrspace(1)* %4, align 4
  %add.ptr2 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr1.sum
  %6 = bitcast i8 addrspace(1)* %add.ptr2 to i32 addrspace(1)*
  %7 = load i32, i32 addrspace(1)* %6, align 4
  %add.ptr3 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 %add.ptr4.sum
  %8 = bitcast i8 addrspace(1)* %add.ptr3 to i32 addrspace(1)*
  %9 = load i32, i32 addrspace(1)* %8, align 4
  %add.ptr6 = getelementptr inbounds i8, i8 addrspace(1)* %main.addr.011, i32 undef
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind }
