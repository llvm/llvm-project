; RUN: opt -mtriple=amdgcn-unknown-amdhsa -S -amdgpu-annotate-kernel-features < %s | FileCheck -check-prefix=HSA %s

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0

; HSA: define void @use_tgid_x(i32 addrspace(1)* %ptr) #1 {
define void @use_tgid_x(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tgid_y(i32 addrspace(1)* %ptr) #2 {
define void @use_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @multi_use_tgid_y(i32 addrspace(1)* %ptr) #2 {
define void @multi_use_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tgid_x_y(i32 addrspace(1)* %ptr) #2 {
define void @use_tgid_x_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tgid_z(i32 addrspace(1)* %ptr) #3 {
define void @use_tgid_z(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tgid_x_z(i32 addrspace(1)* %ptr) #3 {
define void @use_tgid_x_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tgid_y_z(i32 addrspace(1)* %ptr) #4 {
define void @use_tgid_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tgid_x_y_z(i32 addrspace(1)* %ptr) #4 {
define void @use_tgid_x_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val2 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tidig_x(i32 addrspace(1)* %ptr) #1 {
define void @use_tidig_x(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tidig_y(i32 addrspace(1)* %ptr) #5 {
define void @use_tidig_y(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tidig_z(i32 addrspace(1)* %ptr) #6 {
define void @use_tidig_z(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tidig_x_tgid_x(i32 addrspace(1)* %ptr) #1 {
define void @use_tidig_x_tgid_x(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tidig_y_tgid_y(i32 addrspace(1)* %ptr) #7 {
define void @use_tidig_y_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_tidig_x_y_z(i32 addrspace(1)* %ptr) #8 {
define void @use_tidig_x_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workitem.id.y()
  %val2 = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_all_workitems(i32 addrspace(1)* %ptr) #9 {
define void @use_all_workitems(i32 addrspace(1)* %ptr) #1 {
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

; HSA: define void @use_dispatch_ptr(i32 addrspace(1)* %ptr) #10 {
define void @use_dispatch_ptr(i32 addrspace(1)* %ptr) #1 {
  %dispatch.ptr = call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %bc = bitcast i8 addrspace(2)* %dispatch.ptr to i32 addrspace(2)*
  %val = load i32, i32 addrspace(2)* %bc
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

; HSA: attributes #0 = { nounwind readnone }
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
