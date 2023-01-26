; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; CHECK-LABEL: @memset_group_to_flat(
; CHECK: call void @llvm.memset.p3.i64(ptr addrspace(3) align 4 %group.ptr, i8 4, i64 32, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memset_group_to_flat(ptr addrspace(3) %group.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 32, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memset_global_to_flat(
; CHECK: call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %global.ptr, i8 4, i64 32, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memset_global_to_flat(ptr addrspace(1) %global.ptr, i32 %y) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 32, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memset_group_to_flat_no_md(
; CHECK: call void @llvm.memset.p3.i64(ptr addrspace(3) align 4 %group.ptr, i8 4, i64 %size, i1 false){{$}}
define amdgpu_kernel void @memset_group_to_flat_no_md(ptr addrspace(3) %group.ptr, i64 %size) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 %size, i1 false)
  ret void
}

; CHECK-LABEL: @memset_global_to_flat_no_md(
; CHECK: call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %global.ptr, i8 4, i64 %size, i1 false){{$}}
define amdgpu_kernel void @memset_global_to_flat_no_md(ptr addrspace(1) %global.ptr, i64 %size) #0 {
  %cast = addrspacecast ptr addrspace(1) %global.ptr to ptr
  call void @llvm.memset.p0.i64(ptr align 4 %cast, i8 4, i64 %size, i1 false)
  ret void
}

; CHECK-LABEL: @memcpy_flat_to_flat_replace_src_with_group(
; CHECK: call void @llvm.memcpy.p0.p3.i64(ptr align 4 %dest, ptr addrspace(3) align 4 %src.group.ptr, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memcpy_flat_to_flat_replace_src_with_group(ptr %dest, ptr addrspace(3) %src.group.ptr, i64 %size) #0 {
  %cast.src = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %cast.src, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memcpy_inline_flat_to_flat_replace_src_with_group(
; CHECK: call void @llvm.memcpy.inline.p0.p3.i64(ptr align 4 %dest, ptr addrspace(3) align 4 %src.group.ptr, i64 42, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memcpy_inline_flat_to_flat_replace_src_with_group(ptr %dest, ptr addrspace(3) %src.group.ptr) #0 {
  %cast.src = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  call void @llvm.memcpy.inline.p0.p0.i64(ptr align 4 %dest, ptr align 4 %cast.src, i64 42, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memcpy_flat_to_flat_replace_dest_with_group(
; CHECK: call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) align 4 %dest.group.ptr, ptr align 4 %src.ptr, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memcpy_flat_to_flat_replace_dest_with_group(ptr addrspace(3) %dest.group.ptr, ptr %src.ptr, i64 %size) #0 {
  %cast.dest = addrspacecast ptr addrspace(3) %dest.group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %cast.dest, ptr align 4 %src.ptr, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memcpy_flat_to_flat_replace_dest_src_with_group(
; CHECK: call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) align 4 %src.group.ptr, ptr addrspace(3) align 4 %src.group.ptr, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memcpy_flat_to_flat_replace_dest_src_with_group(ptr addrspace(3) %dest.group.ptr, ptr addrspace(3) %src.group.ptr, i64 %size) #0 {
  %cast.src = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  %cast.dest = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %cast.dest, ptr align 4 %cast.src, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memcpy_flat_to_flat_replace_dest_group_src_global(
; CHECK: call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) align 4 %dest.group.ptr, ptr addrspace(1) align 4 %src.global.ptr, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memcpy_flat_to_flat_replace_dest_group_src_global(ptr addrspace(3) %dest.group.ptr, ptr addrspace(1) %src.global.ptr, i64 %size) #0 {
  %cast.src = addrspacecast ptr addrspace(1) %src.global.ptr to ptr
  %cast.dest = addrspacecast ptr addrspace(3) %dest.group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %cast.dest, ptr align 4 %cast.src, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memcpy_group_to_flat_replace_dest_global(
; CHECK: call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %dest.global.ptr, ptr addrspace(3) align 4 %src.group.ptr, i32 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memcpy_group_to_flat_replace_dest_global(ptr addrspace(1) %dest.global.ptr, ptr addrspace(3) %src.group.ptr, i32 %size) #0 {
  %cast.dest = addrspacecast ptr addrspace(1) %dest.global.ptr to ptr
  call void @llvm.memcpy.p0.p3.i32(ptr align 4 %cast.dest, ptr addrspace(3) align 4 %src.group.ptr, i32 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

; CHECK-LABEL: @memcpy_flat_to_flat_replace_src_with_group_tbaa_struct(
; CHECK: call void @llvm.memcpy.p0.p3.i64(ptr align 4 %dest, ptr addrspace(3) align 4 %src.group.ptr, i64 %size, i1 false), !tbaa.struct !8
define amdgpu_kernel void @memcpy_flat_to_flat_replace_src_with_group_tbaa_struct(ptr %dest, ptr addrspace(3) %src.group.ptr, i64 %size) #0 {
  %cast.src = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %cast.src, i64 %size, i1 false), !tbaa.struct !8
  ret void
}

; CHECK-LABEL: @memcpy_flat_to_flat_replace_src_with_group_no_md(
; CHECK: call void @llvm.memcpy.p0.p3.i64(ptr align 4 %dest, ptr addrspace(3) align 4 %src.group.ptr, i64 %size, i1 false){{$}}
define amdgpu_kernel void @memcpy_flat_to_flat_replace_src_with_group_no_md(ptr %dest, ptr addrspace(3) %src.group.ptr, i64 %size) #0 {
  %cast.src = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest, ptr align 4 %cast.src, i64 %size, i1 false)
  ret void
}

; CHECK-LABEL: @multiple_memcpy_flat_to_flat_replace_src_with_group_no_md(
; CHECK: call void @llvm.memcpy.p0.p3.i64(ptr align 4 %dest0, ptr addrspace(3) align 4 %src.group.ptr, i64 %size, i1 false){{$}}
; CHECK: call void @llvm.memcpy.p0.p3.i64(ptr align 4 %dest1, ptr addrspace(3) align 4 %src.group.ptr, i64 %size, i1 false){{$}}
define amdgpu_kernel void @multiple_memcpy_flat_to_flat_replace_src_with_group_no_md(ptr %dest0, ptr %dest1, ptr addrspace(3) %src.group.ptr, i64 %size) #0 {
  %cast.src = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest0, ptr align 4 %cast.src, i64 %size, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dest1, ptr align 4 %cast.src, i64 %size, i1 false)
  ret void
}

; Check for iterator problems if the pointer has 2 uses in the same call
; CHECK-LABEL: @memcpy_group_flat_to_flat_self(
; CHECK: call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) align 4 %group.ptr, ptr addrspace(3) align 4 %group.ptr, i64 32, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memcpy_group_flat_to_flat_self(ptr addrspace(3) %group.ptr) #0 {
  %cast = addrspacecast ptr addrspace(3) %group.ptr to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %cast, ptr align 4 %cast, i64 32, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}
; CHECK-LABEL: @memmove_flat_to_flat_replace_src_with_group(
; CHECK: call void @llvm.memmove.p0.p3.i64(ptr align 4 %dest, ptr addrspace(3) align 4 %src.group.ptr, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
define amdgpu_kernel void @memmove_flat_to_flat_replace_src_with_group(ptr %dest, ptr addrspace(3) %src.group.ptr, i64 %size) #0 {
  %cast.src = addrspacecast ptr addrspace(3) %src.group.ptr to ptr
  call void @llvm.memmove.p0.p0.i64(ptr align 4 %dest, ptr align 4 %cast.src, i64 %size, i1 false), !tbaa !0, !alias.scope !3, !noalias !6
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #1
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1
declare void @llvm.memcpy.inline.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1
declare void @llvm.memcpy.p0.p3.i32(ptr nocapture writeonly, ptr addrspace(3) nocapture readonly, i32, i1) #1
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"A", !2}
!2 = !{!"tbaa root"}
!3 = !{!4}
!4 = distinct !{!4, !5, !"some scope 1"}
!5 = distinct !{!5, !"some domain"}
!6 = !{!7}
!7 = distinct !{!7, !5, !"some scope 2"}
!8 = !{i64 0, i64 8, null}
