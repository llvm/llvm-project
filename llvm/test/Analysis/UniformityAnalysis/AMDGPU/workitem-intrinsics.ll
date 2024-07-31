; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #0

; CHECK: DIVERGENT:  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @workitem_id_x() #1 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK: DIVERGENT:  %id.y = call i32 @llvm.amdgcn.workitem.id.y()
define amdgpu_kernel void @workitem_id_y() #1 {
  %id.y = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %id.y, ptr addrspace(1) undef
  ret void
}

; CHECK: DIVERGENT:  %id.z = call i32 @llvm.amdgcn.workitem.id.z()
define amdgpu_kernel void @workitem_id_z() #1 {
  %id.z = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %id.z, ptr addrspace(1) undef
  ret void
}

; CHECK: DIVERGENT:  %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 0, i32 0)
define amdgpu_kernel void @mbcnt_lo() #1 {
  %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 0, i32 0)
  store volatile i32 %mbcnt.lo, ptr addrspace(1) undef
  ret void
}

; CHECK: DIVERGENT:  %mbcnt.hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 0, i32 0)
define amdgpu_kernel void @mbcnt_hi() #1 {
  %mbcnt.hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 0, i32 0)
  store volatile i32 %mbcnt.hi, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_singlethreaded':
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_x_singlethreaded() #2 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_y_singlethreaded':
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_y_singlethreaded() #2 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_z_singlethreaded':
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_z_singlethreaded() #2 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_singlethreaded_md':
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_x_singlethreaded_md() !reqd_work_group_size !0 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_y_singlethreaded_md':
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_y_singlethreaded_md() !reqd_work_group_size !0 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_z_singlethreaded_md':
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_z_singlethreaded_md() !reqd_work_group_size !0 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_not_singlethreaded_dimx':
; CHECK: DIVERGENT:  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @workitem_id_x_not_singlethreaded_dimx() !reqd_work_group_size !1 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_not_singlethreaded_dimy':
; CHECK: DIVERGENT:  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @workitem_id_x_not_singlethreaded_dimy() !reqd_work_group_size !2 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_not_singlethreaded_dimz':
; CHECK: DIVERGENT:  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @workitem_id_x_not_singlethreaded_dimz() !reqd_work_group_size !3 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id.x, ptr addrspace(1) undef
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { "amdgpu-flat-work-group-size"="1,1" }

!0 = !{i32 1, i32 1, i32 1}
!1 = !{i32 2, i32 1, i32 1}
!2 = !{i32 1, i32 2, i32 1}
!3 = !{i32 1, i32 1, i32 2}
