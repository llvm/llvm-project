; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=amdgpu-lower-kernel-attributes,instcombine,infer-alignment %s | FileCheck -enable-var-scope %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=amdgpu-lower-kernel-attributes,instcombine,infer-alignment %s | FileCheck -enable-var-scope %s

target datalayout = "n32"

; CHECK-LABEL: @invalid_reqd_work_group_size(
; CHECK: load i16,
define amdgpu_kernel void @invalid_reqd_work_group_size(ptr addrspace(1) %out) #0 !reqd_work_group_size !1 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  store i16 %group.size.x, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @volatile_load_group_size_x(
; CHECK: load volatile i16,
define amdgpu_kernel void @volatile_load_group_size_x(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load volatile i16, ptr addrspace(4) %gep.group.size.x, align 4
  store i16 %group.size.x, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @load_group_size_x(
; CHECK: store i16 %group.size.x,
define amdgpu_kernel void @load_group_size_x(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  store i16 %group.size.x, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @load_group_size_y(
; CHECK: store i16 %group.size.y,
define amdgpu_kernel void @load_group_size_y(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.y = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 6
  %group.size.y = load i16, ptr addrspace(4) %gep.group.size.y, align 4
  store i16 %group.size.y, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @load_group_size_z(
; CHECK: store i16 %group.size.z,
define amdgpu_kernel void @load_group_size_z(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.z = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 8
  %group.size.z = load i16, ptr addrspace(4) %gep.group.size.z, align 4
  store i16 %group.size.z, ptr addrspace(1) %out
  ret void
}

; Metadata uses i64 instead of i32
; CHECK-LABEL: @load_group_size_x_reqd_work_group_size_i64(
; CHECK: store i16 %group.size.x,
define amdgpu_kernel void @load_group_size_x_reqd_work_group_size_i64(ptr addrspace(1) %out) #0 !reqd_work_group_size !2 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  store i16 %group.size.x, ptr addrspace(1) %out
  ret void
}

; Metadata uses i16 instead of i32
; CHECK-LABEL: @load_group_size_x_reqd_work_group_size_i16(
; CHECK: store i16 %group.size.x,
define amdgpu_kernel void @load_group_size_x_reqd_work_group_size_i16(ptr addrspace(1) %out) #0 !reqd_work_group_size !3 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  store i16 %group.size.x, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @use_local_size_x_8_16_2(
; CHECK: store i64 %zext,
define amdgpu_kernel void @use_local_size_x_8_16_2(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 12
  %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @use_local_size_y_8_16_2(
; CHECK: store i64 %zext,
define amdgpu_kernel void @use_local_size_y_8_16_2(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.y = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 6
  %group.size.y = load i16, ptr addrspace(4) %gep.group.size.y, align 4
  %gep.grid.size.y = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 16
  %grid.size.y = load i32, ptr addrspace(4) %gep.grid.size.y, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.y()
  %group.size.y.zext = zext i16 %group.size.y to i32
  %group.id_x_group.size.y = mul i32 %group.id, %group.size.y.zext
  %sub = sub i32 %grid.size.y, %group.id_x_group.size.y
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.y.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @use_local_size_z_8_16_2(
; CHECK: store i64 %zext,
define amdgpu_kernel void @use_local_size_z_8_16_2(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.z = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 8
  %group.size.z = load i16, ptr addrspace(4) %gep.group.size.z, align 4
  %gep.grid.size.z = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 20
  %grid.size.z = load i32, ptr addrspace(4) %gep.grid.size.z, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.z()
  %group.size.z.zext = zext i16 %group.size.z to i32
  %group.id_x_group.size.z = mul i32 %group.id, %group.size.z.zext
  %sub = sub i32 %grid.size.z, %group.id_x_group.size.z
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.z.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; Simplification on select is invalid, but we can still eliminate the
; load of the group size.

; CHECK-LABEL: @local_size_x_8_16_2_wrong_group_id(
; CHECK: %group.id = tail call i32 @llvm.amdgcn.workgroup.id.y()
; CHECK: %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
define amdgpu_kernel void @local_size_x_8_16_2_wrong_group_id(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 12
  %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.y()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @local_size_x_8_16_2_wrong_grid_size(
; CHECK: %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
; CHECK: %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  define amdgpu_kernel void @local_size_x_8_16_2_wrong_grid_size(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 16
  %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @local_size_x_8_16_2_wrong_cmp_type(
; CHECK: %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
; CHECK: %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
; CHECK: %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
; CHECK: %smin = call i32 @llvm.smin.i32(i32 %sub, i32 %group.size.x.zext)
define amdgpu_kernel void @local_size_x_8_16_2_wrong_cmp_type(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 12
  %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
  %smin = call i32 @llvm.smin.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %smin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @local_size_x_8_16_2_wrong_select(
; CHECK: %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
; CHECK: %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
; CHECK: %umax = call i32 @llvm.umax.i32(i32 %sub, i32 %group.size.x.zext)
; CHECK: %zext = zext i32 %umax to i64
define amdgpu_kernel void @local_size_x_8_16_2_wrong_select(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 12
  %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
  %umax = call i32 @llvm.umax.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %umax to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @use_local_size_x_8_16_2_wrong_grid_load_size(
; CHECK: %grid.size.x = load i16, ptr addrspace(4) %gep.grid.size.x, align 4
; CHECK: %grid.size.x.zext = zext i16 %grid.size.x to i32
; CHECK: %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
; CHECK: %sub = sub i32 %grid.size.x.zext, %group.id_x_group.size.x
define amdgpu_kernel void @use_local_size_x_8_16_2_wrong_grid_load_size(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 12
  %grid.size.x = load i16, ptr addrspace(4) %gep.grid.size.x, align 4
  %grid.size.x.zext = zext i16 %grid.size.x to i32
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x.zext, %group.id_x_group.size.x
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @func_group_size_x(
; CHECK: ret i32 %zext
define i32 @func_group_size_x(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %zext = zext i16 %group.size.x to i32
  ret i32 %zext
}

; CHECK-LABEL: @__ockl_get_local_size_reqd_size(
; CHECK: %group.size = phi i16 [ %tmp24, %bb17 ], [ %tmp16, %bb9 ], [ %tmp8, %bb1 ], [ 1, %bb ]
define i64 @__ockl_get_local_size_reqd_size(i32 %arg) #1 !reqd_work_group_size !0 {
bb:
  %tmp = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #2
  switch i32 %arg, label %bb25 [
    i32 0, label %bb1
    i32 1, label %bb9
    i32 2, label %bb17
  ]

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %tmp3 = getelementptr inbounds i8, ptr addrspace(4) %tmp, i64 12
  %tmp5 = load i32, ptr addrspace(4) %tmp3, align 4
  %tmp6 = getelementptr inbounds i8, ptr addrspace(4) %tmp, i64 4
  %tmp8 = load i16, ptr addrspace(4) %tmp6, align 4
  br label %bb25

bb9:                                              ; preds = %bb
  %tmp10 = tail call i32 @llvm.amdgcn.workgroup.id.y()
  %tmp11 = getelementptr inbounds i8, ptr addrspace(4) %tmp, i64 16
  %tmp13 = load i32, ptr addrspace(4) %tmp11, align 8
  %tmp14 = getelementptr inbounds i8, ptr addrspace(4) %tmp, i64 6
  %tmp16 = load i16, ptr addrspace(4) %tmp14, align 2
  br label %bb25

bb17:                                             ; preds = %bb
  %tmp18 = tail call i32 @llvm.amdgcn.workgroup.id.z()
  %tmp19 = getelementptr inbounds i8, ptr addrspace(4) %tmp, i64 20
  %tmp21 = load i32, ptr addrspace(4) %tmp19, align 4
  %tmp22 = getelementptr inbounds i8, ptr addrspace(4) %tmp, i64 8
  %tmp24 = load i16, ptr addrspace(4) %tmp22, align 8
  br label %bb25

bb25:                                             ; preds = %bb17, %bb9, %bb1, %bb
  %tmp26 = phi i32 [ %tmp21, %bb17 ], [ %tmp13, %bb9 ], [ %tmp5, %bb1 ], [ 0, %bb ]
  %group.size = phi i16 [ %tmp24, %bb17 ], [ %tmp16, %bb9 ], [ %tmp8, %bb1 ], [ 1, %bb ]
  %tmp28 = phi i32 [ %tmp18, %bb17 ], [ %tmp10, %bb9 ], [ %tmp2, %bb1 ], [ 0, %bb ]
  %tmp29 = zext i16 %group.size to i32
  %tmp30 = mul i32 %tmp28, %tmp29
  %tmp31 = sub i32 %tmp26, %tmp30
  %umin = call i32 @llvm.umin.i32(i32 %tmp31, i32 %tmp29)
  %tmp34 = zext i32 %umin to i64
  ret i64 %tmp34
}

; CHECK-LABEL: @all_local_size(
; CHECK: store volatile i64 %tmp34.i, ptr addrspace(1) %out, align 4
; CHECK-NEXT: store volatile i64 %tmp34.i14, ptr addrspace(1) %out, align 4
; CHECK-NEXT: store volatile i64 %tmp34.i7, ptr addrspace(1) %out, align 4
define amdgpu_kernel void @all_local_size(ptr addrspace(1) nocapture readnone %out) #0 !reqd_work_group_size !0 {
  %tmp.i = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
  %tmp2.i = tail call i32 @llvm.amdgcn.workgroup.id.x() #0
  %tmp3.i = getelementptr inbounds i8, ptr addrspace(4) %tmp.i, i64 12
  %tmp5.i = load i32, ptr addrspace(4) %tmp3.i, align 4
  %tmp6.i = getelementptr inbounds i8, ptr addrspace(4) %tmp.i, i64 4
  %tmp8.i = load i16, ptr addrspace(4) %tmp6.i, align 4
  %tmp29.i = zext i16 %tmp8.i to i32
  %tmp30.i = mul i32 %tmp2.i, %tmp29.i
  %tmp31.i = sub i32 %tmp5.i, %tmp30.i
  %umin0 = call i32 @llvm.umin.i32(i32 %tmp31.i, i32 %tmp29.i)
  %tmp34.i = zext i32 %umin0 to i64
  %tmp10.i = tail call i32 @llvm.amdgcn.workgroup.id.y() #0
  %tmp11.i = getelementptr inbounds i8, ptr addrspace(4) %tmp.i, i64 16
  %tmp13.i = load i32, ptr addrspace(4) %tmp11.i, align 8
  %tmp14.i = getelementptr inbounds i8, ptr addrspace(4) %tmp.i, i64 6
  %tmp16.i = load i16, ptr addrspace(4) %tmp14.i, align 2
  %tmp29.i9 = zext i16 %tmp16.i to i32
  %tmp30.i10 = mul i32 %tmp10.i, %tmp29.i9
  %tmp31.i11 = sub i32 %tmp13.i, %tmp30.i10
  %umin1 = call i32 @llvm.umin.i32(i32 %tmp31.i11, i32 %tmp29.i9)
  %tmp34.i14 = zext i32 %umin1 to i64
  %tmp18.i = tail call i32 @llvm.amdgcn.workgroup.id.z() #0
  %tmp19.i = getelementptr inbounds i8, ptr addrspace(4) %tmp.i, i64 20
  %tmp21.i = load i32, ptr addrspace(4) %tmp19.i, align 4
  %tmp22.i = getelementptr inbounds i8, ptr addrspace(4) %tmp.i, i64 8
  %tmp24.i = load i16, ptr addrspace(4) %tmp22.i, align 8
  %tmp29.i2 = zext i16 %tmp24.i to i32
  %tmp30.i3 = mul i32 %tmp18.i, %tmp29.i2
  %tmp31.i4 = sub i32 %tmp21.i, %tmp30.i3
  %umin2 = call i32 @llvm.umin.i32(i32 %tmp31.i4, i32 %tmp29.i2)
  %tmp34.i7 = zext i32 %umin2 to i64
  store volatile i64 %tmp34.i, ptr addrspace(1) %out, align 4
  store volatile i64 %tmp34.i14, ptr addrspace(1) %out, align 4
  store volatile i64 %tmp34.i7, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: Should be able to handle this, but not much reason to.
; CHECK-LABEL: @partial_load_group_size_x(
; CHECK-NEXT: %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT: %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
; CHECK-NEXT: %group.size.x.lo = load i8, ptr addrspace(4) %gep.group.size.x, align 4
; CHECK-NEXT: store i8 %group.size.x.lo, ptr addrspace(1) %out, align 1
define amdgpu_kernel void @partial_load_group_size_x(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x.lo = load i8, ptr addrspace(4) %gep.group.size.x, align 1
  store i8 %group.size.x.lo, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @partial_load_group_size_x_explicit_callsite_align(
; CHECK-NEXT: %dispatch.ptr = tail call align 2 ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT: %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
; CHECK-NEXT: %group.size.x.lo = load i8, ptr addrspace(4) %gep.group.size.x, align 2
; CHECK-NEXT: store i8 %group.size.x.lo, ptr addrspace(1) %out, align 1
define amdgpu_kernel void @partial_load_group_size_x_explicit_callsite_align(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call align 2 ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x.lo = load i8, ptr addrspace(4) %gep.group.size.x, align 1
  store i8 %group.size.x.lo, ptr addrspace(1) %out
  ret void
}

; TODO: Should be able to handle this
; CHECK-LABEL: @load_group_size_xy_i32(
; CHECK: %group.size.xy = load i32,
; CHECK: store i32 %group.size.xy
define amdgpu_kernel void @load_group_size_xy_i32(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.xy = load i32, ptr addrspace(4) %gep.group.size.x, align 4
  store i32 %group.size.xy, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @load_group_size_x_y_multiple_dispatch_ptr(
; CHECK: store volatile i16 %group.size.x, ptr addrspace(1) %out, align 2
; CHECK: store volatile i16 %group.size.y, ptr addrspace(1) %out, align 2
define amdgpu_kernel void @load_group_size_x_y_multiple_dispatch_ptr(ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
  %dispatch.ptr0 = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr0, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  store volatile i16 %group.size.x, ptr addrspace(1) %out

  %dispatch.ptr1 = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.y = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr1, i64 6
  %group.size.y = load i16, ptr addrspace(4) %gep.group.size.y, align 4
  store volatile i16 %group.size.y, ptr addrspace(1) %out

  ret void
}

; CHECK-LABEL: @use_local_size_x_uniform_work_group_size(
; CHECK-NEXT: %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT: %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
; CHECK-NEXT: %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
; CHECK: %group.size.x.zext = zext i16 %group.size.x to i32
; CHECK: store i64 %zext, ptr addrspace(1) %out
define amdgpu_kernel void @use_local_size_x_uniform_work_group_size(ptr addrspace(1) %out) #2 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 12
  %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @use_local_size_x_uniform_work_group_size_false(
; CHECK: call i32 @llvm.umin
define amdgpu_kernel void @use_local_size_x_uniform_work_group_size_false(ptr addrspace(1) %out) #3 {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %gep.group.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 4
  %group.size.x = load i16, ptr addrspace(4) %gep.group.size.x, align 4
  %gep.grid.size.x = getelementptr inbounds i8, ptr addrspace(4) %dispatch.ptr, i64 12
  %grid.size.x = load i32, ptr addrspace(4) %gep.grid.size.x, align 4
  %group.id = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %group.size.x.zext = zext i16 %group.size.x to i32
  %group.id_x_group.size.x = mul i32 %group.id, %group.size.x.zext
  %sub = sub i32 %grid.size.x, %group.id_x_group.size.x
  %umin = call i32 @llvm.umin.i32(i32 %sub, i32 %group.size.x.zext)
  %zext = zext i32 %umin to i64
  store i64 %zext, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @no_use_dispatch_ptr(
; CHECK-NEXT: ret void
define amdgpu_kernel void @no_use_dispatch_ptr() {
  %dispatch.ptr = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  ret void
}

declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #1
declare i32 @llvm.amdgcn.workgroup.id.x() #1
declare i32 @llvm.amdgcn.workgroup.id.y() #1
declare i32 @llvm.amdgcn.workgroup.id.z() #1
declare i32 @llvm.umin.i32(i32, i32) #1
declare i32 @llvm.smin.i32(i32, i32) #1
declare i32 @llvm.umax.i32(i32, i32) #1

attributes #0 = { nounwind "uniform-work-group-size"="true" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind "uniform-work-group-size"="true" }
attributes #3 = { nounwind "uniform-work-group-size"="false" }

!0 = !{i32 8, i32 16, i32 2}
!1 = !{i32 8, i32 16}
!2 = !{i64 8, i64 16, i64 2}
!3 = !{i16 8, i16 16, i16 2}

!llvm.module.flags = !{!4}
!4 = !{i32 1, !"amdhsa_code_object_version", i32 500}
