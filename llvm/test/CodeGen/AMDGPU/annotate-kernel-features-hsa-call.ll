; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-annotate-kernel-features %s | FileCheck -check-prefix=HSA %s

declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0
declare i8 addrspace(2)* @llvm.amdgcn.queue.ptr() #0
declare i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr() #0
declare i64 @llvm.amdgcn.dispatch.id() #0

; HSA: define void @use_workitem_id_y() #1 {
define void @use_workitem_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workitem_id_z() #2 {
define void @use_workitem_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workgroup_id_y() #3 {
define void @use_workgroup_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workgroup_id_z() #4 {
define void @use_workgroup_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @use_dispatch_ptr() #5 {
define void @use_dispatch_ptr() #1 {
  %dispatch.ptr = call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  store volatile i8 addrspace(2)* %dispatch.ptr, i8 addrspace(2)* addrspace(1)* undef
  ret void
}

; HSA: define void @use_queue_ptr() #6 {
define void @use_queue_ptr() #1 {
  %queue.ptr = call i8 addrspace(2)* @llvm.amdgcn.queue.ptr()
  store volatile i8 addrspace(2)* %queue.ptr, i8 addrspace(2)* addrspace(1)* undef
  ret void
}

; HSA: define void @use_dispatch_id() #7 {
define void @use_dispatch_id() #1 {
  %val = call i64 @llvm.amdgcn.dispatch.id()
  store volatile i64 %val, i64 addrspace(1)* undef
  ret void
}

; HSA: define void @use_workgroup_id_y_workgroup_id_z() #8 {
define void @use_workgroup_id_y_workgroup_id_z() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* undef
  store volatile i32 %val1, i32 addrspace(1)* undef
  ret void
}

; HSA: define void @func_indirect_use_workitem_id_y() #1 {
define void @func_indirect_use_workitem_id_y() #1 {
  call void @use_workitem_id_y()
  ret void
}

; HSA: define void @func_indirect_use_workitem_id_z() #2 {
define void @func_indirect_use_workitem_id_z() #1 {
  call void @use_workitem_id_z()
  ret void
}

; HSA: define void @func_indirect_use_workgroup_id_y() #3 {
define void @func_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; HSA: define void @func_indirect_use_workgroup_id_z() #4 {
define void @func_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; HSA: define void @func_indirect_indirect_use_workgroup_id_y() #3 {
define void @func_indirect_indirect_use_workgroup_id_y() #1 {
  call void @func_indirect_use_workgroup_id_y()
  ret void
}

; HSA: define void @indirect_x2_use_workgroup_id_y() #3 {
define void @indirect_x2_use_workgroup_id_y() #1 {
  call void @func_indirect_indirect_use_workgroup_id_y()
  ret void
}

; HSA: define void @func_indirect_use_dispatch_ptr() #5 {
define void @func_indirect_use_dispatch_ptr() #1 {
  call void @use_dispatch_ptr()
  ret void
}

; HSA: define void @func_indirect_use_queue_ptr() #6 {
define void @func_indirect_use_queue_ptr() #1 {
  call void @use_queue_ptr()
  ret void
}

; HSA: define void @func_indirect_use_dispatch_id() #7 {
define void @func_indirect_use_dispatch_id() #1 {
  call void @use_dispatch_id()
  ret void
}

; HSA: define void @func_indirect_use_workgroup_id_y_workgroup_id_z() #9 {
define void @func_indirect_use_workgroup_id_y_workgroup_id_z() #1 {
  call void @func_indirect_use_workgroup_id_y_workgroup_id_z()
  ret void
}

; HSA: define void @recursive_use_workitem_id_y() #1 {
define void @recursive_use_workitem_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val, i32 addrspace(1)* undef
  call void @recursive_use_workitem_id_y()
  ret void
}

; HSA: define void @call_recursive_use_workitem_id_y() #1 {
define void @call_recursive_use_workitem_id_y() #1 {
  call void @recursive_use_workitem_id_y()
  ret void
}

; HSA: define void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #6 {
define void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #1 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %stof
  ret void
}

; HSA: define void @use_group_to_flat_addrspacecast_gfx9(i32 addrspace(3)* %ptr) #10 {
define void @use_group_to_flat_addrspacecast_gfx9(i32 addrspace(3)* %ptr) #2 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %stof
  ret void
}

; HSA: define void @use_group_to_flat_addrspacecast_queue_ptr_gfx9(i32 addrspace(3)* %ptr) #11 {
define void @use_group_to_flat_addrspacecast_queue_ptr_gfx9(i32 addrspace(3)* %ptr) #2 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %stof
  call void @func_indirect_use_queue_ptr()
  ret void
}

; HSA: define void @indirect_use_group_to_flat_addrspacecast() #6 {
define void @indirect_use_group_to_flat_addrspacecast() #1 {
  call void @use_group_to_flat_addrspacecast(i32 addrspace(3)* null)
  ret void
}

; HSA: define void @indirect_use_group_to_flat_addrspacecast_gfx9() #9 {
define void @indirect_use_group_to_flat_addrspacecast_gfx9() #1 {
  call void @use_group_to_flat_addrspacecast_gfx9(i32 addrspace(3)* null)
  ret void
}

; HSA: define void @indirect_use_group_to_flat_addrspacecast_queue_ptr_gfx9() #6 {
define void @indirect_use_group_to_flat_addrspacecast_queue_ptr_gfx9() #1 {
  call void @use_group_to_flat_addrspacecast_queue_ptr_gfx9(i32 addrspace(3)* null)
  ret void
}

; HSA: define void @use_kernarg_segment_ptr() #12 {
define void @use_kernarg_segment_ptr() #1 {
  %kernarg.segment.ptr = call i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr()
  store volatile i8 addrspace(2)* %kernarg.segment.ptr, i8 addrspace(2)* addrspace(1)* undef
  ret void
}

; HSA: define void @func_indirect_use_kernarg_segment_ptr() #12 {
define void @func_indirect_use_kernarg_segment_ptr() #1 {
  call void @use_kernarg_segment_ptr()
  ret void
}

; HSA: define void @use_implicitarg_ptr() #12 {
define void @use_implicitarg_ptr() #1 {
  %implicitarg.ptr = call i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr()
  store volatile i8 addrspace(2)* %implicitarg.ptr, i8 addrspace(2)* addrspace(1)* undef
  ret void
}

; HSA: define void @func_indirect_use_implicitarg_ptr() #12 {
define void @func_indirect_use_implicitarg_ptr() #1 {
  call void @use_implicitarg_ptr()
  ret void
}

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind "target-cpu"="fiji" }
attributes #2 = { nounwind "target-cpu"="gfx900" }

; HSA: attributes #0 = { nounwind readnone speculatable }
; HSA: attributes #1 = { nounwind "amdgpu-work-item-id-y" "target-cpu"="fiji" }
; HSA: attributes #2 = { nounwind "amdgpu-work-item-id-z" "target-cpu"="fiji" }
; HSA: attributes #3 = { nounwind "amdgpu-work-group-id-y" "target-cpu"="fiji" }
; HSA: attributes #4 = { nounwind "amdgpu-work-group-id-z" "target-cpu"="fiji" }
; HSA: attributes #5 = { nounwind "amdgpu-dispatch-ptr" "target-cpu"="fiji" }
; HSA: attributes #6 = { nounwind "amdgpu-queue-ptr" "target-cpu"="fiji" }
; HSA: attributes #7 = { nounwind "amdgpu-dispatch-id" "target-cpu"="fiji" }
; HSA: attributes #8 = { nounwind "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "target-cpu"="fiji" }
; HSA: attributes #9 = { nounwind "target-cpu"="fiji" }
; HSA: attributes #10 = { nounwind "target-cpu"="gfx900" }
; HSA: attributes #11 = { nounwind "amdgpu-queue-ptr" "target-cpu"="gfx900" }
; HSA: attributes #12 = { nounwind "amdgpu-kernarg-segment-ptr" "target-cpu"="fiji" }
