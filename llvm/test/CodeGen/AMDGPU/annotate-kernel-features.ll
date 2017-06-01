; RUN: opt -S -mtriple=amdgcn-unknown-unknown -amdgpu-annotate-kernel-features < %s | FileCheck -check-prefix=NOHSA -check-prefix=ALL %s

declare i32 @llvm.r600.read.tgid.x() #0
declare i32 @llvm.r600.read.tgid.y() #0
declare i32 @llvm.r600.read.tgid.z() #0

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.tidig.y() #0
declare i32 @llvm.r600.read.tidig.z() #0

declare i32 @llvm.r600.read.local.size.x() #0
declare i32 @llvm.r600.read.local.size.y() #0
declare i32 @llvm.r600.read.local.size.z() #0

; ALL: define void @use_tgid_x(i32 addrspace(1)* %ptr) #1 {
define void @use_tgid_x(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.tgid.x()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tgid_y(i32 addrspace(1)* %ptr) #2 {
define void @use_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.tgid.y()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @multi_use_tgid_y(i32 addrspace(1)* %ptr) #2 {
define void @multi_use_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tgid.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  %val1 = call i32 @llvm.r600.read.tgid.y()
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tgid_x_y(i32 addrspace(1)* %ptr) #2 {
define void @use_tgid_x_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tgid.x()
  %val1 = call i32 @llvm.r600.read.tgid.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tgid_z(i32 addrspace(1)* %ptr) #3 {
define void @use_tgid_z(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.tgid.z()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tgid_x_z(i32 addrspace(1)* %ptr) #3 {
define void @use_tgid_x_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tgid.x()
  %val1 = call i32 @llvm.r600.read.tgid.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tgid_y_z(i32 addrspace(1)* %ptr) #4 {
define void @use_tgid_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tgid.y()
  %val1 = call i32 @llvm.r600.read.tgid.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tgid_x_y_z(i32 addrspace(1)* %ptr) #4 {
define void @use_tgid_x_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tgid.x()
  %val1 = call i32 @llvm.r600.read.tgid.y()
  %val2 = call i32 @llvm.r600.read.tgid.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tidig_x(i32 addrspace(1)* %ptr) #1 {
define void @use_tidig_x(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.tidig.x()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tidig_y(i32 addrspace(1)* %ptr) #5 {
define void @use_tidig_y(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.tidig.y()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tidig_z(i32 addrspace(1)* %ptr) #6 {
define void @use_tidig_z(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.tidig.z()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tidig_x_tgid_x(i32 addrspace(1)* %ptr) #1 {
define void @use_tidig_x_tgid_x(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tidig.x()
  %val1 = call i32 @llvm.r600.read.tgid.x()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tidig_y_tgid_y(i32 addrspace(1)* %ptr) #7 {
define void @use_tidig_y_tgid_y(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tidig.y()
  %val1 = call i32 @llvm.r600.read.tgid.y()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_tidig_x_y_z(i32 addrspace(1)* %ptr) #8 {
define void @use_tidig_x_y_z(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tidig.x()
  %val1 = call i32 @llvm.r600.read.tidig.y()
  %val2 = call i32 @llvm.r600.read.tidig.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  ret void
}

; ALL: define void @use_all_workitems(i32 addrspace(1)* %ptr) #9 {
define void @use_all_workitems(i32 addrspace(1)* %ptr) #1 {
  %val0 = call i32 @llvm.r600.read.tidig.x()
  %val1 = call i32 @llvm.r600.read.tidig.y()
  %val2 = call i32 @llvm.r600.read.tidig.z()
  %val3 = call i32 @llvm.r600.read.tgid.x()
  %val4 = call i32 @llvm.r600.read.tgid.y()
  %val5 = call i32 @llvm.r600.read.tgid.z()
  store volatile i32 %val0, i32 addrspace(1)* %ptr
  store volatile i32 %val1, i32 addrspace(1)* %ptr
  store volatile i32 %val2, i32 addrspace(1)* %ptr
  store volatile i32 %val3, i32 addrspace(1)* %ptr
  store volatile i32 %val4, i32 addrspace(1)* %ptr
  store volatile i32 %val5, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_get_local_size_x(i32 addrspace(1)* %ptr) #10 {
; NOHSA: define void @use_get_local_size_x(i32 addrspace(1)* %ptr) #1 {
define void @use_get_local_size_x(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.local.size.x()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_get_local_size_y(i32 addrspace(1)* %ptr) #10 {
; NOHSA: define void @use_get_local_size_y(i32 addrspace(1)* %ptr) #1 {
define void @use_get_local_size_y(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.local.size.y()
  store i32 %val, i32 addrspace(1)* %ptr
  ret void
}

; HSA: define void @use_get_local_size_z(i32 addrspace(1)* %ptr) #10 {
; NOHSA: define void @use_get_local_size_z(i32 addrspace(1)* %ptr) #1 {
define void @use_get_local_size_z(i32 addrspace(1)* %ptr) #1 {
  %val = call i32 @llvm.r600.read.local.size.z()
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
