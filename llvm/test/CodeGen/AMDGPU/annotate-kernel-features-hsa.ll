; RUN: opt -mtriple=amdgcn-unknown-amdhsa -S -amdgpu-annotate-kernel-features < %s | FileCheck -check-prefix=HSA %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

declare i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0

declare i1 @llvm.amdgcn.is.shared(i8* nocapture) #2
declare i1 @llvm.amdgcn.is.private(i8* nocapture) #2

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

; HSA: define amdgpu_kernel void @use_is_shared(i8* %ptr) #11 {
define amdgpu_kernel void @use_is_shared(i8* %ptr) #1 {
  %is.shared = call i1 @llvm.amdgcn.is.shared(i8* %ptr)
  %ext = zext i1 %is.shared to i32
  store i32 %ext, i32 addrspace(1)* undef
  ret void
}

; HSA: define amdgpu_kernel void @use_is_private(i8* %ptr) #11 {
define amdgpu_kernel void @use_is_private(i8* %ptr) #1 {
  %is.private = call i1 @llvm.amdgcn.is.private(i8* %ptr)
  %ext = zext i1 %is.private to i32
  store i32 %ext, i32 addrspace(1)* undef
  ret void
}

; HSA: define amdgpu_kernel void @use_alloca() #13 {
define amdgpu_kernel void @use_alloca() #1 {
  %alloca = alloca i32, addrspace(5)
  store i32 0, i32 addrspace(5)* %alloca
  ret void
}

; HSA: define amdgpu_kernel void @use_alloca_non_entry_block() #13 {
define amdgpu_kernel void @use_alloca_non_entry_block() #1 {
entry:
  br label %bb

bb:
  %alloca = alloca i32, addrspace(5)
  store i32 0, i32 addrspace(5)* %alloca
  ret void
}

; HSA: define void @use_alloca_func() #13 {
define void @use_alloca_func() #1 {
  %alloca = alloca i32, addrspace(5)
  store i32 0, i32 addrspace(5)* %alloca
  ret void
}

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }

; HSA: attributes #0 = { nounwind readnone speculatable willreturn }
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
; HSA: attributes #13 = { nounwind "amdgpu-stack-objects" }
