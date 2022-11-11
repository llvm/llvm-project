; RUN: llc < %s | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

; The call to intrinsic implicitarg_ptr reaches a load through a phi. The
; offsets of the phi cannot be determined, and hence the attirbutor assumes that
; hostcall is in use.

; CHECK: .value_kind:     hidden_hostcall_buffer
; CHECK: .value_kind:     hidden_multigrid_sync_arg

define amdgpu_kernel void @the_kernel(i32 addrspace(1)* %a, i64 %index1, i64 %index2, i1 %cond)  {
entry:
  %tmp7 = tail call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  br i1 %cond, label %old, label %new

old:                                              ; preds = %entry
  %tmp4 = getelementptr i8, i8 addrspace(4)* %tmp7, i64 %index1
  br label %join

new:                                              ; preds = %entry
  %tmp12 = getelementptr inbounds i8, i8 addrspace(4)* %tmp7, i64 %index2
  br label %join

join:                                             ; preds = %new, %old
  %.in.in.in = phi i8 addrspace(4)* [ %tmp12, %new ], [ %tmp4, %old ]
  %.in.in = bitcast i8 addrspace(4)* %.in.in.in to i16 addrspace(4)*

  ;;; THIS USE is where the offset into implicitarg_ptr is unknown
  %.in = load i16, i16 addrspace(4)* %.in.in, align 2

  %idx.ext = sext i16 %.in to i64
  %add.ptr3 = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %idx.ext
  %tmp16 = atomicrmw add i32 addrspace(1)* %add.ptr3, i32 15 syncscope("agent-one-as") monotonic, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

declare align 4 i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()

declare i32 @llvm.amdgcn.workgroup.id.x()
