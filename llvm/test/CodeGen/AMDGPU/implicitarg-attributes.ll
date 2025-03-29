; RUN: opt -passes=amdgpu-attributor < %s | llc | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

; The call to intrinsic implicitarg_ptr reaches a load through a phi. The
; offsets of the phi cannot be determined, and hence the attirbutor assumes that
; hostcall is in use.

; CHECK-LABEL: amdhsa.kernels:
; CHECK: .value_kind:     hidden_hostcall_buffer
; CHECK: .value_kind:     hidden_multigrid_sync_arg
; CHECK-LABEL: .name:           kernel_1

define amdgpu_kernel void @kernel_1(ptr addrspace(1) %a, i64 %index1, i64 %index2, i1 %cond)  {
entry:
  %tmp7 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  br i1 %cond, label %old, label %new

old:                                              ; preds = %entry
  %tmp4 = getelementptr i8, ptr addrspace(4) %tmp7, i64 %index1
  br label %join

new:                                              ; preds = %entry
  %tmp12 = getelementptr inbounds i8, ptr addrspace(4) %tmp7, i64 %index2
  br label %join

join:                                             ; preds = %new, %old
  %.in.in.in = phi ptr addrspace(4) [ %tmp12, %new ], [ %tmp4, %old ]

  ;;; THIS USE is where the offset into implicitarg_ptr is unknown
  %.in = load i16, ptr addrspace(4) %.in.in.in, align 2

  %idx.ext = sext i16 %.in to i64
  %add.ptr3 = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %idx.ext
  %tmp16 = atomicrmw add ptr addrspace(1) %add.ptr3, i32 15 syncscope("agent-one-as") monotonic, align 4
  ret void
}

; The call to intrinsic implicitarg_ptr is combined with an offset produced by
; select'ing between two constants, before it is eventually used in a GEP to
; form the address of a load. This test ensures that AAPointerInfo can look
; through the select to maintain a set of indices, so that it can precisely
; determine that hostcall and other expensive implicit args are not in use.

; CHECK-NOT: hidden_hostcall_buffer
; CHECK-NOT: hidden_multigrid_sync_arg
; CHECK-LABEL: .name:           kernel_2

define amdgpu_kernel void @kernel_2(ptr addrspace(1) %a, i1 %cond)  {
entry:
  %tmp7 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %tmp5 = select i1 %cond, i64 12, i64 18
  %tmp6 = getelementptr inbounds i8, ptr addrspace(4) %tmp7, i64 %tmp5

  ;;; THIS USE is where multiple offsets are possible, relative to implicitarg_ptr
  %tmp9 = load i16, ptr addrspace(4) %tmp6, align 2

  %idx.ext = sext i16 %tmp9 to i64
  %add.ptr3 = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %idx.ext
  %tmp16 = atomicrmw add ptr addrspace(1) %add.ptr3, i32 15 syncscope("agent-one-as") monotonic, align 4
  ret void
}

; CHECK-NOT: hidden_hostcall_buffer
; CHECK-NOT: hidden_multigrid_sync_arg
; CHECK-LABEL: .name:           kernel_3

define amdgpu_kernel void @kernel_3(ptr addrspace(1) %a, i1 %cond)  {
entry:
  %tmp7 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  br i1 %cond, label %old, label %new

old:                                              ; preds = %entry
  %tmp4 = getelementptr i8, ptr addrspace(4) %tmp7, i64 12
  br label %join

new:                                              ; preds = %entry
  %tmp12 = getelementptr inbounds i8, ptr addrspace(4) %tmp7, i64 18
  br label %join

join:                                             ; preds = %new, %old
  %.in.in.in = phi ptr addrspace(4) [ %tmp12, %new ], [ %tmp4, %old ]

  ;;; THIS USE of implicitarg_ptr should not produce hostcall metadata
  %.in = load i16, ptr addrspace(4) %.in.in.in, align 2

  %idx.ext = sext i16 %.in to i64
  %add.ptr3 = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %idx.ext
  %tmp16 = atomicrmw add ptr addrspace(1) %add.ptr3, i32 15 syncscope("agent-one-as") monotonic, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

declare align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()

declare i32 @llvm.amdgcn.workgroup.id.x()
