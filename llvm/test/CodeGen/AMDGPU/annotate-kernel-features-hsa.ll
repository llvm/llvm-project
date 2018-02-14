; RUN: opt -mtriple=amdgcn-unknown-amdhsa -S -amdgpu-annotate-kernel-features < %s | FileCheck -check-prefix=HSA %s

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

declare i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0

; HSA: define amdgpu_kernel void @use_tgid_x(i32 addrspace(1)* %ptr) #1 {
define amdgpu_kernel void @use_tgid_x(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tgid_y(i32 addrspace(1)* %ptr) #2 {
define amdgpu_kernel void @use_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @multi_use_tgid_y(i32 addrspace(1)* %ptr) #2 {
define amdgpu_kernel void @multi_use_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tgid_x_y(i32 addrspace(1)* %ptr) #2 {
define amdgpu_kernel void @use_tgid_x_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tgid_z(i32 addrspace(1)* %ptr) #3 {
define amdgpu_kernel void @use_tgid_z(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tgid_x_z(i32 addrspace(1)* %ptr) #3 {
define amdgpu_kernel void @use_tgid_x_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tgid_y_z(i32 addrspace(1)* %ptr) #4 {
define amdgpu_kernel void @use_tgid_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tgid_x_y_z(i32 addrspace(1)* %ptr) #4 {
define amdgpu_kernel void @use_tgid_x_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val2 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tidig_x(i32 addrspace(1)* %ptr) #1 {
define amdgpu_kernel void @use_tidig_x(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tidig_y(i32 addrspace(1)* %ptr) #5 {
define amdgpu_kernel void @use_tidig_y(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tidig_z(i32 addrspace(1)* %ptr) #6 {
define amdgpu_kernel void @use_tidig_z(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tidig_x_tgid_x(i32 addrspace(1)* %ptr) #1 {
define amdgpu_kernel void @use_tidig_x_tgid_x(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tidig_y_tgid_y(i32 addrspace(1)* %ptr) #7 {
define amdgpu_kernel void @use_tidig_y_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_tidig_x_y_z(i32 addrspace(1)* %ptr) #8 {
define amdgpu_kernel void @use_tidig_x_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workitem.id.y()
  %val2 = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_all_workitems(i32 addrspace(1)* %ptr) #9 {
define amdgpu_kernel void @use_all_workitems(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workitem.id.y()
  %val2 = call i32 @llvm.amdgcn.workitem.id.z()
  %val3 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val4 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val5 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  store volatile i32 %val3, i32 addrspace(1)* %ptr
  store volatile i32 %val4, i32 addrspace(1)* %ptr
  store volatile i32 %val5, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_dispatch_ptr(i32 addrspace(1)* %ptr) #10 {
define amdgpu_kernel void @use_dispatch_ptr(i32 addrspace(1)* %ptr) #1 {
  %dispatch.ptr = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %bc = bitcast i8 addrspace(4)* %dispatch.ptr to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %bc
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_queue_ptr(i32 addrspace(1)* %ptr) #11 {
define amdgpu_kernel void @use_queue_ptr(i32 addrspace(1)* %ptr) #1 {
  %dispatch.ptr = call i8 addrspace(4)* @llvm.amdgcn.queue.ptr()
  %bc = bitcast i8 addrspace(4)* %dispatch.ptr to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %bc
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_kernarg_segment_ptr(i32 addrspace(1)* %ptr) #12 {
define amdgpu_kernel void @use_kernarg_segment_ptr(i32 addrspace(1)* %ptr) #1 {
  %dispatch.ptr = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %bc = bitcast i8 addrspace(4)* %dispatch.ptr to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %bc
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define amdgpu_kernel void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #11 {
define amdgpu_kernel void @use_group_to_flat_addrspacecast(i32 addrspace(3)* %ptr) #1 {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32*
  store volatile i32 0, i32* %stof
  ret void
}

; HSA: define amdgpu_kernel void @use_private_to_flat_addrspacecast(i32 addrspace(5)* %ptr) #11 {
define amdgpu_kernel void @use_private_to_flat_addrspacecast(i32 addrspace(5)* %ptr) #1 {
  %stof = addrspacecast i32 addrspace(5)* %ptr to i32*
  store volatile i32 0, i32* %stof
  ret void
}

; HSA: define amdgpu_kernel void @use_flat_to_group_addrspacecast(i32* %ptr) #1 {
define amdgpu_kernel void @use_flat_to_group_addrspacecast(i32* %ptr) #1 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(3)*
  store volatile i32 0, i32 addrspace(3)* %ftos
  ret void
}

; HSA: define amdgpu_kernel void @use_flat_to_private_addrspacecast(i32* %ptr) #1 {
define amdgpu_kernel void @use_flat_to_private_addrspacecast(i32* %ptr) #1 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(5)*
  store volatile i32 0, i32 addrspace(5)* %ftos
  ret void
}

; No-op addrspacecast should not use queue ptr
; HSA: define amdgpu_kernel void @use_global_to_flat_addrspacecast(i32 addrspace(1)* %ptr) #1 {
define amdgpu_kernel void @use_global_to_flat_addrspacecast(i32 addrspace(1)* %ptr) #1 {
  %stof = addrspacecast i32 addrspace(1)* %ptr to i32*
  store volatile i32 0, i32* %stof
  ret void
}

; HSA: define amdgpu_kernel void @use_constant_to_flat_addrspacecast(i32 addrspace(4)* %ptr) #1 {
define amdgpu_kernel void @use_constant_to_flat_addrspacecast(i32 addrspace(4)* %ptr) #1 {
  %stof = addrspacecast i32 addrspace(4)* %ptr to i32*
  %ld = load volatile i32, i32* %stof
  ret void
}

; HSA: define amdgpu_kernel void @use_flat_to_global_addrspacecast(i32* %ptr) #1 {
define amdgpu_kernel void @use_flat_to_global_addrspacecast(i32* %ptr) #1 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(1)*
  store volatile i32 0, i32 addrspace(1)* %ftos
  ret void
}

; HSA: define amdgpu_kernel void @use_flat_to_constant_addrspacecast(i32* %ptr) #1 {
define amdgpu_kernel void @use_flat_to_constant_addrspacecast(i32* %ptr) #1 {
  %ftos = addrspacecast i32* %ptr to i32 addrspace(4)*
  %ld = load volatile i32, i32 addrspace(4)* %ftos
  ret void
}

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }

; HSA: attributes #0 = { nounwind readnone speculatable }
; HSA: attributes #1 = { nounwind }
; HSA: attributes #2 = { nounwind "amdgpu-work-group-id-y" }
; HSA: attributes #3 = { nounwind "amdgpu-work-group-id-z" }
; HSA: attributes #4 = { nounwind "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" }
; HSA: attributes #5 = { nounwind "amdgpu-work-item-id-y" }
; HSA: attributes #6 = { nounwind "amdgpu-work-item-id-z" }
; HSA: attributes #7 = { nounwind "amdgpu-work-group-id-y" "amdgpu-work-item-id-y" }
; HSA: attributes #8 = { nounwind "amdgpu-work-item-id-y" "amdgpu-work-item-id-z" }
; HSA: attributes #9 = { nounwind "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "amdgpu-work-item-id-y" "amdgpu-work-item-id-z" }
; HSA: attributes #10 = { nounwind "amdgpu-dispatch-ptr" }
; HSA: attributes #11 = { nounwind "amdgpu-queue-ptr" }
; HSA: attributes #12 = { nounwind "amdgpu-kernarg-segment-ptr" }
