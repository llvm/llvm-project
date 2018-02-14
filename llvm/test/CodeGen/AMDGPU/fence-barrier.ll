; RUN: llc -mtriple=amdgcn-amd-amdhsa-amdgiz -mcpu=gfx803 -enable-si-insert-waitcnts=1 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llvm-as -data-layout=A5 < %s | llc -mtriple=amdgcn-amd-amdhsa-amdgiz -mcpu=gfx803 -enable-si-insert-waitcnts=1 -verify-machineinstrs | FileCheck --check-prefix=GCN %s

declare i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
declare i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
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
define amdgpu_kernel void @test_local(i32 addrspace(1)*) {
  %2 = alloca i32 addrspace(1)*, align 4, addrspace(5)
  store i32 addrspace(1)* %0, i32 addrspace(1)* addrspace(5)* %2, align 4
  %3 = call i32 @llvm.amdgcn.workitem.id.x()
  %4 = zext i32 %3 to i64
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %6, label %7

; <label>:6:                                      ; preds = %1
  store i32 1911, i32 addrspace(3)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(3)* @test_local.temp, i64 0, i64 0), align 4
  br label %7

; <label>:7:                                      ; preds = %6, %1
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %8 = load i32, i32 addrspace(3)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(3)* @test_local.temp, i64 0, i64 0), align 4
  %9 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %2, align 4
  %10 = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %11 = call i32 @llvm.amdgcn.workitem.id.x()
  %12 = call i32 @llvm.amdgcn.workgroup.id.x()
  %13 = getelementptr inbounds i8, i8 addrspace(4)* %10, i64 4
  %14 = bitcast i8 addrspace(4)* %13 to i16 addrspace(4)*
  %15 = load i16, i16 addrspace(4)* %14, align 4
  %16 = zext i16 %15 to i32
  %17 = mul i32 %12, %16
  %18 = add i32 %17, %11
  %19 = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %20 = zext i32 %18 to i64
  %21 = bitcast i8 addrspace(4)* %19 to i64 addrspace(4)*
  %22 = load i64, i64 addrspace(4)* %21, align 8
  %23 = add i64 %22, %20
  %24 = getelementptr inbounds i32, i32 addrspace(1)* %9, i64 %23
  store i32 %8, i32 addrspace(1)* %24, align 4
  ret void
}

; GCN-LABEL: {{^}}test_global
; GCN: v_add_u32_e32 v{{[0-9]+}}, vcc, 0x888, v{{[0-9]+}}
; GCN: flat_store_dword
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN-NEXT: s_barrier
define amdgpu_kernel void @test_global(i32 addrspace(1)*) {
  %2 = alloca i32 addrspace(1)*, align 4, addrspace(5)
  %3 = alloca i32, align 4, addrspace(5)
  store i32 addrspace(1)* %0, i32 addrspace(1)* addrspace(5)* %2, align 4
  store i32 0, i32 addrspace(5)* %3, align 4
  br label %4

; <label>:4:                                      ; preds = %58, %1
  %5 = load i32, i32 addrspace(5)* %3, align 4
  %6 = sext i32 %5 to i64
  %7 = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %8 = call i32 @llvm.amdgcn.workitem.id.x()
  %9 = call i32 @llvm.amdgcn.workgroup.id.x()
  %10 = getelementptr inbounds i8, i8 addrspace(4)* %7, i64 4
  %11 = bitcast i8 addrspace(4)* %10 to i16 addrspace(4)*
  %12 = load i16, i16 addrspace(4)* %11, align 4
  %13 = zext i16 %12 to i32
  %14 = mul i32 %9, %13
  %15 = add i32 %14, %8
  %16 = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %17 = zext i32 %15 to i64
  %18 = bitcast i8 addrspace(4)* %16 to i64 addrspace(4)*
  %19 = load i64, i64 addrspace(4)* %18, align 8
  %20 = add i64 %19, %17
  %21 = icmp ult i64 %6, %20
  br i1 %21, label %22, label %61

; <label>:22:                                     ; preds = %4
  %23 = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %24 = call i32 @llvm.amdgcn.workitem.id.x()
  %25 = call i32 @llvm.amdgcn.workgroup.id.x()
  %26 = getelementptr inbounds i8, i8 addrspace(4)* %23, i64 4
  %27 = bitcast i8 addrspace(4)* %26 to i16 addrspace(4)*
  %28 = load i16, i16 addrspace(4)* %27, align 4
  %29 = zext i16 %28 to i32
  %30 = mul i32 %25, %29
  %31 = add i32 %30, %24
  %32 = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %33 = zext i32 %31 to i64
  %34 = bitcast i8 addrspace(4)* %32 to i64 addrspace(4)*
  %35 = load i64, i64 addrspace(4)* %34, align 8
  %36 = add i64 %35, %33
  %37 = add i64 %36, 2184
  %38 = trunc i64 %37 to i32
  %39 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %2, align 4
  %40 = load i32, i32 addrspace(5)* %3, align 4
  %41 = sext i32 %40 to i64
  %42 = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %43 = call i32 @llvm.amdgcn.workitem.id.x()
  %44 = call i32 @llvm.amdgcn.workgroup.id.x()
  %45 = getelementptr inbounds i8, i8 addrspace(4)* %42, i64 4
  %46 = bitcast i8 addrspace(4)* %45 to i16 addrspace(4)*
  %47 = load i16, i16 addrspace(4)* %46, align 4
  %48 = zext i16 %47 to i32
  %49 = mul i32 %44, %48
  %50 = add i32 %49, %43
  %51 = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %52 = zext i32 %50 to i64
  %53 = bitcast i8 addrspace(4)* %51 to i64 addrspace(4)*
  %54 = load i64, i64 addrspace(4)* %53, align 8
  %55 = add i64 %54, %52
  %56 = add i64 %41, %55
  %57 = getelementptr inbounds i32, i32 addrspace(1)* %39, i64 %56
  store i32 %38, i32 addrspace(1)* %57, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  br label %58

; <label>:58:                                     ; preds = %22
  %59 = load i32, i32 addrspace(5)* %3, align 4
  %60 = add nsw i32 %59, 1
  store i32 %60, i32 addrspace(5)* %3, align 4
  br label %4

; <label>:61:                                     ; preds = %4
  ret void
}

; GCN-LABEL: {{^}}test_global_local
; GCN: v_mov_b32_e32 v[[VAL:[0-9]+]], 0x999
; GCN: ds_write_b32 v{{[0-9]+}}, v[[VAL]]
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN-NEXT: s_barrier
; GCN: flat_store_dword
define amdgpu_kernel void @test_global_local(i32 addrspace(1)*) {
  %2 = alloca i32 addrspace(1)*, align 4, addrspace(5)
  store i32 addrspace(1)* %0, i32 addrspace(1)* addrspace(5)* %2, align 4
  %3 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %2, align 4
  %4 = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %5 = call i32 @llvm.amdgcn.workitem.id.x()
  %6 = call i32 @llvm.amdgcn.workgroup.id.x()
  %7 = getelementptr inbounds i8, i8 addrspace(4)* %4, i64 4
  %8 = bitcast i8 addrspace(4)* %7 to i16 addrspace(4)*
  %9 = load i16, i16 addrspace(4)* %8, align 4
  %10 = zext i16 %9 to i32
  %11 = mul i32 %6, %10
  %12 = add i32 %11, %5
  %13 = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %14 = zext i32 %12 to i64
  %15 = bitcast i8 addrspace(4)* %13 to i64 addrspace(4)*
  %16 = load i64, i64 addrspace(4)* %15, align 8
  %17 = add i64 %16, %14
  %18 = getelementptr inbounds i32, i32 addrspace(1)* %3, i64 %17
  store i32 1, i32 addrspace(1)* %18, align 4
  %19 = call i32 @llvm.amdgcn.workitem.id.x()
  %20 = zext i32 %19 to i64
  %21 = icmp eq i64 %20, 0
  br i1 %21, label %22, label %23

; <label>:22:                                     ; preds = %1
  store i32 2457, i32 addrspace(3)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(3)* @test_global_local.temp, i64 0, i64 0), align 4
  br label %23

; <label>:23:                                     ; preds = %22, %1
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %24 = load i32, i32 addrspace(3)* getelementptr inbounds ([1 x i32], [1 x i32] addrspace(3)* @test_global_local.temp, i64 0, i64 0), align 4
  %25 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %2, align 4
  %26 = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %27 = call i32 @llvm.amdgcn.workitem.id.x()
  %28 = call i32 @llvm.amdgcn.workgroup.id.x()
  %29 = getelementptr inbounds i8, i8 addrspace(4)* %26, i64 4
  %30 = bitcast i8 addrspace(4)* %29 to i16 addrspace(4)*
  %31 = load i16, i16 addrspace(4)* %30, align 4
  %32 = zext i16 %31 to i32
  %33 = mul i32 %28, %32
  %34 = add i32 %33, %27
  %35 = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %36 = zext i32 %34 to i64
  %37 = bitcast i8 addrspace(4)* %35 to i64 addrspace(4)*
  %38 = load i64, i64 addrspace(4)* %37, align 8
  %39 = add i64 %38, %36
  %40 = getelementptr inbounds i32, i32 addrspace(1)* %25, i64 %39
  store i32 %24, i32 addrspace(1)* %40, align 4
  ret void
}
