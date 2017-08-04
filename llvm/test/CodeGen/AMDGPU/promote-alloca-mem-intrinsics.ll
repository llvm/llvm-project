; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-promote-alloca < %s | FileCheck %s

declare void @llvm.memcpy.p0i8.p1i8.i32(i8* nocapture, i8 addrspace(1)* nocapture, i32, i32, i1) #0
declare void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* nocapture, i8* nocapture, i32, i32, i1) #0

declare void @llvm.memmove.p0i8.p1i8.i32(i8* nocapture, i8 addrspace(1)* nocapture, i32, i32, i1) #0
declare void @llvm.memmove.p1i8.p0i8.i32(i8 addrspace(1)* nocapture, i8* nocapture, i32, i32, i1) #0

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) #0

declare i32 @llvm.objectsize.i32.p0i8(i8*, i1, i1) #1

; CHECK-LABEL: @promote_with_memcpy(
; CHECK: getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_memcpy.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* %alloca.bc, i8 addrspace(1)* %in.bc, i32 68, i32 4, i1 false)
; CHECK: call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* %out.bc, i8 addrspace(3)* %alloca.bc, i32 68, i32 4, i1 false)
define amdgpu_kernel void @promote_with_memcpy(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %in.bc = bitcast i32 addrspace(1)* %in to i8 addrspace(1)*
  %out.bc = bitcast i32 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* %alloca.bc, i8 addrspace(1)* %in.bc, i32 68, i32 4, i1 false)
  call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* %out.bc, i8* %alloca.bc, i32 68, i32 4, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_memmove(
; CHECK: getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_memmove.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memmove.p3i8.p1i8.i32(i8 addrspace(3)* %alloca.bc, i8 addrspace(1)* %in.bc, i32 68, i32 4, i1 false)
; CHECK: call void @llvm.memmove.p1i8.p3i8.i32(i8 addrspace(1)* %out.bc, i8 addrspace(3)* %alloca.bc, i32 68, i32 4, i1 false)
define amdgpu_kernel void @promote_with_memmove(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %in.bc = bitcast i32 addrspace(1)* %in to i8 addrspace(1)*
  %out.bc = bitcast i32 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memmove.p0i8.p1i8.i32(i8* %alloca.bc, i8 addrspace(1)* %in.bc, i32 68, i32 4, i1 false)
  call void @llvm.memmove.p1i8.p0i8.i32(i8 addrspace(1)* %out.bc, i8* %alloca.bc, i32 68, i32 4, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_memset(
; CHECK: getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_memset.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memset.p3i8.i32(i8 addrspace(3)* %alloca.bc, i8 7, i32 68, i32 4, i1 false)
define amdgpu_kernel void @promote_with_memset(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %in.bc = bitcast i32 addrspace(1)* %in to i8 addrspace(1)*
  %out.bc = bitcast i32 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memset.p0i8.i32(i8* %alloca.bc, i8 7, i32 68, i32 4, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_objectsize(
; CHECK: [[PTR:%[0-9]+]] = getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_objectsize.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call i32 @llvm.objectsize.i32.p3i8(i8 addrspace(3)* %alloca.bc, i1 false, i1 false)
define amdgpu_kernel void @promote_with_objectsize(i32 addrspace(1)* %out) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %size = call i32 @llvm.objectsize.i32.p0i8(i8* %alloca.bc, i1 false, i1 false)
  store i32 %size, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="64,64" "amdgpu-waves-per-eu"="1,3" }
attributes #1 = { nounwind readnone }
