; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llvm-as -data-layout=A5 < %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs | FileCheck --check-prefix=GCN %s

declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare void @llvm.amdgcn.s.barrier()

@test_local.temp = internal addrspace(3) global [1 x i32] undef, align 4
@test_global_local.temp = internal addrspace(3) global [1 x i32] undef, align 4


; GCN-LABEL: {{^}}test_local
; GCN: v_mov_b32_e32 v[[VAL:[0-9]+]], 0x777
; GCN: ds_write_b32 v{{[0-9]+}}, v[[VAL]]
; GCN: s_waitcnt lgkmcnt(0){{$}}
; GCN-NEXT: s_barrier
; GCN: flat_store_dword
define amdgpu_kernel void @test_local(ptr addrspace(1) %arg) {
bb:
  %i = alloca ptr addrspace(1), align 4, addrspace(5)
  store ptr addrspace(1) %arg, ptr addrspace(5) %i, align 4
  %i1 = call i32 @llvm.amdgcn.workitem.id.x()
  %i2 = zext i32 %i1 to i64
  %i3 = icmp eq i64 %i2, 0
  br i1 %i3, label %bb4, label %bb5

bb4:                                              ; preds = %bb
  store i32 1911, ptr addrspace(3) @test_local.temp, align 4
  br label %bb5

bb5:                                              ; preds = %bb4, %bb
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %i6 = load i32, ptr addrspace(3) @test_local.temp, align 4
  %i7 = load ptr addrspace(1), ptr addrspace(5) %i, align 4
  %i8 = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %i9 = call i32 @llvm.amdgcn.workitem.id.x()
  %i10 = call i32 @llvm.amdgcn.workgroup.id.x()
  %i11 = getelementptr inbounds i8, ptr addrspace(4) %i8, i64 4
  %i13 = load i16, ptr addrspace(4) %i11, align 4
  %i14 = zext i16 %i13 to i32
  %i15 = mul i32 %i10, %i14
  %i16 = add i32 %i15, %i9
  %i17 = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %i18 = zext i32 %i16 to i64
  %i20 = load i64, ptr addrspace(4) %i17, align 8
  %i21 = add i64 %i20, %i18
  %i22 = getelementptr inbounds i32, ptr addrspace(1) %i7, i64 %i21
  store i32 %i6, ptr addrspace(1) %i22, align 4
  ret void
}

; GCN-LABEL: {{^}}test_global
; GCN: v_add_u32_e32 v{{[0-9]+}}, vcc, 0x888, v{{[0-9]+}}
; GCN: flat_store_dword
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN-NEXT: s_barrier
define amdgpu_kernel void @test_global(ptr addrspace(1) %arg) {
bb:
  %i = alloca ptr addrspace(1), align 4, addrspace(5)
  %i1 = alloca i32, align 4, addrspace(5)
  store ptr addrspace(1) %arg, ptr addrspace(5) %i, align 4
  store i32 0, ptr addrspace(5) %i1, align 4
  br label %bb2

bb2:                                              ; preds = %bb56, %bb
  %i3 = load i32, ptr addrspace(5) %i1, align 4
  %i4 = sext i32 %i3 to i64
  %i5 = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %i6 = call i32 @llvm.amdgcn.workitem.id.x()
  %i7 = call i32 @llvm.amdgcn.workgroup.id.x()
  %i8 = getelementptr inbounds i8, ptr addrspace(4) %i5, i64 4
  %i10 = load i16, ptr addrspace(4) %i8, align 4
  %i11 = zext i16 %i10 to i32
  %i12 = mul i32 %i7, %i11
  %i13 = add i32 %i12, %i6
  %i14 = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %i15 = zext i32 %i13 to i64
  %i17 = load i64, ptr addrspace(4) %i14, align 8
  %i18 = add i64 %i17, %i15
  %i19 = icmp ult i64 %i4, %i18
  br i1 %i19, label %bb20, label %bb59

bb20:                                             ; preds = %bb2
  %i21 = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %i22 = call i32 @llvm.amdgcn.workitem.id.x()
  %i23 = call i32 @llvm.amdgcn.workgroup.id.x()
  %i24 = getelementptr inbounds i8, ptr addrspace(4) %i21, i64 4
  %i26 = load i16, ptr addrspace(4) %i24, align 4
  %i27 = zext i16 %i26 to i32
  %i28 = mul i32 %i23, %i27
  %i29 = add i32 %i28, %i22
  %i30 = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %i31 = zext i32 %i29 to i64
  %i33 = load i64, ptr addrspace(4) %i30, align 8
  %i34 = add i64 %i33, %i31
  %i35 = add i64 %i34, 2184
  %i36 = trunc i64 %i35 to i32
  %i37 = load ptr addrspace(1), ptr addrspace(5) %i, align 4
  %i38 = load i32, ptr addrspace(5) %i1, align 4
  %i39 = sext i32 %i38 to i64
  %i40 = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %i41 = call i32 @llvm.amdgcn.workitem.id.x()
  %i42 = call i32 @llvm.amdgcn.workgroup.id.x()
  %i43 = getelementptr inbounds i8, ptr addrspace(4) %i40, i64 4
  %i45 = load i16, ptr addrspace(4) %i43, align 4
  %i46 = zext i16 %i45 to i32
  %i47 = mul i32 %i42, %i46
  %i48 = add i32 %i47, %i41
  %i49 = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %i50 = zext i32 %i48 to i64
  %i52 = load i64, ptr addrspace(4) %i49, align 8
  %i53 = add i64 %i52, %i50
  %i54 = add i64 %i39, %i53
  %i55 = getelementptr inbounds i32, ptr addrspace(1) %i37, i64 %i54
  store i32 %i36, ptr addrspace(1) %i55, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  br label %bb56

bb56:                                             ; preds = %bb20
  %i57 = load i32, ptr addrspace(5) %i1, align 4
  %i58 = add nsw i32 %i57, 1
  store i32 %i58, ptr addrspace(5) %i1, align 4
  br label %bb2

bb59:                                             ; preds = %bb2
  ret void
}

; GCN-LABEL: {{^}}test_global_local
; GCN: v_mov_b32_e32 v[[VAL:[0-9]+]], 0x999
; GCN: ds_write_b32 v{{[0-9]+}}, v[[VAL]]
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN-NEXT: s_barrier
; GCN: flat_store_dword
define amdgpu_kernel void @test_global_local(ptr addrspace(1) %arg) {
bb:
  %i = alloca ptr addrspace(1), align 4, addrspace(5)
  store ptr addrspace(1) %arg, ptr addrspace(5) %i, align 4
  %i1 = load ptr addrspace(1), ptr addrspace(5) %i, align 4
  %i2 = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %i3 = call i32 @llvm.amdgcn.workitem.id.x()
  %i4 = call i32 @llvm.amdgcn.workgroup.id.x()
  %i5 = getelementptr inbounds i8, ptr addrspace(4) %i2, i64 4
  %i7 = load i16, ptr addrspace(4) %i5, align 4
  %i8 = zext i16 %i7 to i32
  %i9 = mul i32 %i4, %i8
  %i10 = add i32 %i9, %i3
  %i11 = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %i12 = zext i32 %i10 to i64
  %i14 = load i64, ptr addrspace(4) %i11, align 8
  %i15 = add i64 %i14, %i12
  %i16 = getelementptr inbounds i32, ptr addrspace(1) %i1, i64 %i15
  store i32 1, ptr addrspace(1) %i16, align 4
  %i17 = call i32 @llvm.amdgcn.workitem.id.x()
  %i18 = zext i32 %i17 to i64
  %i19 = icmp eq i64 %i18, 0
  br i1 %i19, label %bb20, label %bb21

bb20:                                             ; preds = %bb
  store i32 2457, ptr addrspace(3) @test_global_local.temp, align 4
  br label %bb21

bb21:                                             ; preds = %bb20, %bb
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %i22 = load i32, ptr addrspace(3) @test_global_local.temp, align 4
  %i23 = load ptr addrspace(1), ptr addrspace(5) %i, align 4
  %i24 = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %i25 = call i32 @llvm.amdgcn.workitem.id.x()
  %i26 = call i32 @llvm.amdgcn.workgroup.id.x()
  %i27 = getelementptr inbounds i8, ptr addrspace(4) %i24, i64 4
  %i29 = load i16, ptr addrspace(4) %i27, align 4
  %i30 = zext i16 %i29 to i32
  %i31 = mul i32 %i26, %i30
  %i32 = add i32 %i31, %i25
  %i33 = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %i34 = zext i32 %i32 to i64
  %i36 = load i64, ptr addrspace(4) %i33, align 8
  %i37 = add i64 %i36, %i34
  %i38 = getelementptr inbounds i32, ptr addrspace(1) %i23, i64 %i37
  store i32 %i22, ptr addrspace(1) %i38, align 4
  ret void
}
