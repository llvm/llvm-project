; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-annotate-kernel-features %s | FileCheck -check-prefix=HSA %s

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

declare i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
declare i64 @llvm.amdgcn.dispatch.id() #0

; HSA: define void @use_workitem_id_x() #1 {
define void @use_workitem_id_x() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workitem_id_y() #2 {
define void @use_workitem_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workitem_id_z() #3 {
define void @use_workitem_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workgroup_id_x() #4 {
define void @use_workgroup_id_x() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workgroup_id_y() #5 {
define void @use_workgroup_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workgroup_id_z() #6 {
define void @use_workgroup_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_dispatch_ptr() #7 {
define void @use_dispatch_ptr() #1 {
  %dispatch.ptr = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  store volatile i8 addrspace(4)* %dispatch.ptr, i8 addrspace(4)* addrspace(1)* undef
  ret void
}

; HSA: define void @use_queue_ptr() #8 {
define void @use_queue_ptr() #1 {
  %queue.ptr = call i8 addrspace(4)* @llvm.amdgcn.queue.ptr()
  store volatile i8 addrspace(4)* %queue.ptr, i8 addrspace(4)* addrspace(1)* undef
  ret void
}

; HSA: define void @use_dispatch_id() #9 {
define void @use_dispatch_id() #1 {
  %val = call i64 @llvm.amdgcn.dispatch.id()
  store volatile i64 %val, i64 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workgroup_id_y_workgroup_id_z() #10 {
define void @use_workgroup_id_y_workgroup_id_z() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* undef
  store volatile i32 %val1, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @func_indirect_use_workitem_id_x() #1 {
define void @func_indirect_use_workitem_id_x() #1 {
  call void @use_workitem_id_x()
  ret void
}

; HSA: define void @kernel_indirect_use_workitem_id_x() #1 {
define void @kernel_indirect_use_workitem_id_x() #1 {
  call void @use_workitem_id_x()
  ret void
}

; HSA: define void @func_indirect_use_workitem_id_y() #2 {
define void @func_indirect_use_workitem_id_y() #1 {
  call void @use_workitem_id_y()
  ret void
}

; HSA: define void @func_indirect_use_workitem_id_z() #3 {
define void @func_indirect_use_workitem_id_z() #1 {
  call void @use_workitem_id_z()
  ret void
}

; HSA: define void @func_indirect_use_workgroup_id_x() #4 {
define void @func_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; HSA: define void @kernel_indirect_use_workgroup_id_x() #4 {
define void @kernel_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; HSA: define void @func_indirect_use_workgroup_id_y() #5 {
define void @func_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; HSA: define void @func_indirect_use_workgroup_id_z() #6 {
define void @func_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; HSA: define void @func_indirect_indirect_use_workgroup_id_y() #5 {
define void @func_indirect_indirect_use_workgroup_id_y() #1 {
  call void @func_indirect_use_workgroup_id_y()
  ret void
}

; HSA: define void @indirect_x2_use_workgroup_id_y() #5 {
define void @indirect_x2_use_workgroup_id_y() #1 {
  call void @func_indirect_indirect_use_workgroup_id_y()
  ret void
}

; HSA: define void @func_indirect_use_dispatch_ptr() #7 {
define void @func_indirect_use_dispatch_ptr() #1 {
  call void @use_dispatch_ptr()
  ret void
}

; HSA: define void @func_indirect_use_queue_ptr() #8 {
define void @func_indirect_use_queue_ptr() #1 {
  call void @use_queue_ptr()
  ret void
}

; HSA: define void @func_indirect_use_dispatch_id() #9 {
define void @func_indirect_use_dispatch_id() #1 {
  call void @use_dispatch_id()
  ret void
}

; HSA: define void @func_indirect_use_workgroup_id_y_workgroup_id_z() #11 {
define void @func_indirect_use_workgroup_id_y_workgroup_id_z() #1 {
  call void @func_indirect_use_workgroup_id_y_workgroup_id_z()
  ret void
}

; HSA: define void @recursive_use_workitem_id_y() #2 {
define void @recursive_use_workitem_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val, i32 addrspace(1)* undef
  call void @recursive_use_workitem_id_y()
  ret void
}

; HSA: define void @call_recursive_use_workitem_id_y() #2 {
define void @call_recursive_use_workitem_id_y() #1 {
  call void @recursive_use_workitem_id_y()
  ret void
}

; HSA: define void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #8 {
define void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #1 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %stof
  ret void
}

; HSA: define void @use_group_to_flat_addrspacecast_gfx9(i32 addrspace(3)* %ptr) #12 {
define void @use_group_to_flat_addrspacecast_gfx9(i32 addrspace(3)* %ptr) #2 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %stof
  ret void
}

; HSA: define void @use_group_to_flat_addrspacecast_queue_ptr_gfx9(i32 addrspace(3)* %ptr) #13 {
define void @use_group_to_flat_addrspacecast_queue_ptr_gfx9(i32 addrspace(3)* %ptr) #2 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %stof
  call void @func_indirect_use_queue_ptr()
  ret void
}

; HSA: define void @indirect_use_group_to_flat_addrspacecast() #8 {
define void @indirect_use_group_to_flat_addrspacecast() #1 {
  call void @use_group_to_flat_addrspacecast(i32 addrspace(3)* null)
  ret void
}

; HSA: define void @indirect_use_group_to_flat_addrspacecast_gfx9() #11 {
define void @indirect_use_group_to_flat_addrspacecast_gfx9() #1 {
  call void @use_group_to_flat_addrspacecast_gfx9(i32 addrspace(3)* null)
  ret void
}

; HSA: define void @indirect_use_group_to_flat_addrspacecast_queue_ptr_gfx9() #8 {
define void @indirect_use_group_to_flat_addrspacecast_queue_ptr_gfx9() #1 {
  call void @use_group_to_flat_addrspacecast_queue_ptr_gfx9(i32 addrspace(3)* null)
  ret void
}

; HSA: define void @use_kernarg_segment_ptr() #14 {
define void @use_kernarg_segment_ptr() #1 {
  %kernarg.segment.ptr = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  store volatile i8 addrspace(4)* %kernarg.segment.ptr, i8 addrspace(4)* addrspace(1)* undef
  ret void
}

; HSA: define void @func_indirect_use_kernarg_segment_ptr() #11 {
define void @func_indirect_use_kernarg_segment_ptr() #1 {
  call void @use_kernarg_segment_ptr()
  ret void
}

; HSA: define amdgpu_kernel void @kern_use_implicitarg_ptr() #15 {
define amdgpu_kernel void @kern_use_implicitarg_ptr() #1 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  store volatile i8 addrspace(4)* %implicitarg.ptr, i8 addrspace(4)* addrspace(1)* undef
  ret void
}

; HSA: define void @use_implicitarg_ptr() #16 {
define void @use_implicitarg_ptr() #1 {
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  store volatile i8 addrspace(4)* %implicitarg.ptr, i8 addrspace(4)* addrspace(1)* undef
  ret void
}

; HSA: define void @func_indirect_use_implicitarg_ptr() #16 {
define void @func_indirect_use_implicitarg_ptr() #1 {
  call void @use_implicitarg_ptr()
  ret void
}

; HSA: declare void @external.func() #17
declare void @external.func() #3

; HSA: define internal void @defined.func() #17 {
define internal void @defined.func() #3 {
  ret void
}

; HSA: define void @func_call_external() #17 {
define void @func_call_external() #3 {
  call void @external.func()
  ret void
}

; HSA: define void @func_call_defined() #17 {
define void @func_call_defined() #3 {
  call void @defined.func()
  ret void
}

; HSA: define void @func_call_asm() #18 {
define void @func_call_asm() #3 {
  call void asm sideeffect "", ""() #3
  ret void
}

; HSA: define amdgpu_kernel void @kern_call_external() #19 {
define amdgpu_kernel void @kern_call_external() #3 {
  call void @external.func()
  ret void
}

; HSA: define amdgpu_kernel void @func_kern_defined() #19 {
define amdgpu_kernel void @func_kern_defined() #3 {
  call void @defined.func()
  ret void
}

; HSA: define i32 @use_dispatch_ptr_ret_type() #20 {
define i32 @use_dispatch_ptr_ret_type() #1 {
  %dispatch.ptr = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  store volatile i8 addrspace(4)* %dispatch.ptr, i8 addrspace(4)* addrspace(1)* undef
  ret i32 0
}

; HSA: define float @func_indirect_use_dispatch_ptr_constexpr_cast_func() #20 {
define float @func_indirect_use_dispatch_ptr_constexpr_cast_func() #1 {
  %f = call float bitcast (i32()* @use_dispatch_ptr_ret_type to float()*)()
  %fadd = fadd float %f, 1.0
  ret float %fadd
}

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind "target-cpu"="fiji" }
attributes #2 = { nounwind "target-cpu"="gfx900" }
attributes #3 = { nounwind }

; HSA: attributes #0 = { nounwind readnone speculatable willreturn }
; HSA: attributes #1 = { nounwind "amdgpu-work-item-id-x" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #2 = { nounwind "amdgpu-work-item-id-y" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #3 = { nounwind "amdgpu-work-item-id-z" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #4 = { nounwind "amdgpu-work-group-id-x" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #5 = { nounwind "amdgpu-work-group-id-y" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #6 = { nounwind "amdgpu-work-group-id-z" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #7 = { nounwind "amdgpu-dispatch-ptr" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #8 = { nounwind "amdgpu-queue-ptr" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #9 = { nounwind "amdgpu-dispatch-id" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #10 = { nounwind "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "target-cpu"="fiji" }
; HSA: attributes #11 = { nounwind "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #12 = { nounwind "target-cpu"="gfx900" "uniform-work-group-size"="false" }
; HSA: attributes #13 = { nounwind "amdgpu-queue-ptr" "target-cpu"="gfx900" "uniform-work-group-size"="false" }
; HSA: attributes #14 = { nounwind "amdgpu-kernarg-segment-ptr" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #15 = { nounwind "amdgpu-implicitarg-ptr" "target-cpu"="fiji" }
; HSA: attributes #16 = { nounwind "amdgpu-implicitarg-ptr" "target-cpu"="fiji" "uniform-work-group-size"="false" }
; HSA: attributes #17 = { nounwind "uniform-work-group-size"="false" }
; HSA: attributes #18 = { nounwind }
; HSA: attributes #19 = { nounwind "amdgpu-calls" "uniform-work-group-size"="false" }
; HSA: attributes #20 = { nounwind "amdgpu-dispatch-ptr" "target-cpu"="fiji" }
