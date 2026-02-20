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

; CHECK-LABEL: UniformityInfo for function 'workitem_id_z_uniform_len_1'
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_z_uniform_len_1(ptr %o) !reqd_work_group_size !4 {
  %id.z = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %id.z, ptr %o
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_div_wavefront_size'
; CHECK: DIVERGENT: %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_x_div_wavefront_size(ptr %o) #3 !reqd_work_group_size !5 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.sg = lshr i32 %id.x, 6
  store i32 %id.sg, ptr %o
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_masked'
; CHECK: DIVERGENT: %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_x_masked(ptr %o) #3 !reqd_work_group_size !5 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.sg.shl.6 = and i32 %id.x, 192
  store i32 %id.sg.shl.6, ptr %o
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_zext_masked'
; CHECK: DIVERGENT: %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT: %id.x.zext = zext nneg i32 %id.x to i64
; CHECK: DIVERGENT: %my.out = getelementptr i64, ptr %o, i64 %id.x.zext
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: store i64 %id.sg.shl.6, ptr %my.out
define amdgpu_kernel void @workitem_id_x_zext_masked(ptr %o) #3 !reqd_work_group_size !5 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.x.zext = zext nneg i32 %id.x to i64
  %my.out = getelementptr i64, ptr %o, i64 %id.x.zext
  %id.sg.shl.6 = and i64 %id.x.zext, 192
  store i64 %id.sg.shl.6, ptr %my.out
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_div_wavefront_size_masked'
; CHECK: DIVERGENT: %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT: %id.x.masked = and i32 %id.x, 127
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_x_div_wavefront_size_masked(ptr %o) #3 !reqd_work_group_size !5 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.x.masked = and i32 %id.x, 127
  %id.sg = lshr i32 %id.x.masked, 6
  store i32 %id.sg, ptr %o
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_x_div_wavefront_size_trunc_masked'
; CHECK: DIVERGENT: %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT: %id.x.trunc = trunc nuw nsw i32 %id.x to i16
; CHECK: DIVERGENT: %id.x.masked = and i16 %id.x.trunc, 127
; CHECK: DIVERGENT: %offset = zext nneg i16 %id.x.masked to i64
; CHECK: DIVERGENT: %my.out = getelementptr i16, ptr %o, i64 %offset
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: store i16 %id.sg, ptr %my.out
define amdgpu_kernel void @workitem_id_x_div_wavefront_size_trunc_masked(ptr %o) #3 !reqd_work_group_size !5 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.x.trunc = trunc nuw nsw i32 %id.x to i16
  %id.x.masked = and i16 %id.x.trunc, 127
  %offset = zext nneg i16 %id.x.masked to i64
  %my.out = getelementptr i16, ptr %o, i64 %offset
  %id.sg = lshr i16 %id.x.masked, 6
  store i16 %id.sg, ptr %my.out
  ret void
}

; CHECK-LABEL: UniformityInfo for function 'workitem_id_y_uniform_in_subgroup'
; CHECK-NOT: DIVERGENT
define amdgpu_kernel void @workitem_id_y_uniform_in_subgroup(ptr %o) #3 !reqd_work_group_size !5 {
  %id.y = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %id.y, ptr %o
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { "amdgpu-flat-work-group-size"="1,1" }
attributes #3 = { "target-cpu"="gfx900" "amdgpu-flat-work-group-size"="256,256" }

!0 = !{i32 1, i32 1, i32 1}
!1 = !{i32 2, i32 1, i32 1}
!2 = !{i32 1, i32 2, i32 1}
!3 = !{i32 1, i32 1, i32 2}
!4 = !{i32 64, i32 1, i32 1}
!5 = !{i32 128, i32 2, i32 1}
!6 = !{i32 256, i32 1, i32 1}
