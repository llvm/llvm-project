; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -amdgpu-promote-alloca < %s | FileCheck %s

declare void @llvm.memcpy.p0i8.p1i8.i32(i8* nocapture, i8 addrspace(1)* nocapture, i32, i1) #0
declare void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* nocapture, i8* nocapture, i32, i1) #0
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) #0

declare void @llvm.memmove.p0i8.p1i8.i32(i8* nocapture, i8 addrspace(1)* nocapture, i32, i1) #0
declare void @llvm.memmove.p1i8.p0i8.i32(i8 addrspace(1)* nocapture, i8* nocapture, i32, i1) #0
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) #0

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) #0

declare i32 @llvm.objectsize.i32.p0i8(i8*, i1, i1, i1) #1

; CHECK-LABEL: @promote_with_memcpy(
; CHECK: getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_memcpy.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %alloca.bc, i8 addrspace(1)* align 4 %in.bc, i32 68, i1 false)
; CHECK: call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out.bc, i8 addrspace(3)* align 4 %alloca.bc, i32 68, i1 false)
define amdgpu_kernel void @promote_with_memcpy(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %in.bc = bitcast i32 addrspace(1)* %in to i8 addrspace(1)*
  %out.bc = bitcast i32 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* align 4 %alloca.bc, i8 addrspace(1)* align 4 %in.bc, i32 68, i1 false)
  call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* align 4 %out.bc, i8* align 4 %alloca.bc, i32 68, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_memmove(
; CHECK: getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_memmove.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memmove.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %alloca.bc, i8 addrspace(1)* align 4 %in.bc, i32 68, i1 false)
; CHECK: call void @llvm.memmove.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out.bc, i8 addrspace(3)* align 4 %alloca.bc, i32 68, i1 false)
define amdgpu_kernel void @promote_with_memmove(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %in.bc = bitcast i32 addrspace(1)* %in to i8 addrspace(1)*
  %out.bc = bitcast i32 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memmove.p0i8.p1i8.i32(i8* align 4 %alloca.bc, i8 addrspace(1)* align 4 %in.bc, i32 68, i1 false)
  call void @llvm.memmove.p1i8.p0i8.i32(i8 addrspace(1)* align 4 %out.bc, i8* align 4 %alloca.bc, i32 68, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_memset(
; CHECK: getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_memset.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memset.p3i8.i32(i8 addrspace(3)* align 4 %alloca.bc, i8 7, i32 68, i1 false)
define amdgpu_kernel void @promote_with_memset(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %in.bc = bitcast i32 addrspace(1)* %in to i8 addrspace(1)*
  %out.bc = bitcast i32 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memset.p0i8.i32(i8* align 4 %alloca.bc, i8 7, i32 68, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_objectsize(
; CHECK: [[PTR:%[0-9]+]] = getelementptr inbounds [64 x [17 x i32]], [64 x [17 x i32]] addrspace(3)* @promote_with_objectsize.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call i32 @llvm.objectsize.i32.p3i8(i8 addrspace(3)* %alloca.bc, i1 false, i1 false, i1 false)
define amdgpu_kernel void @promote_with_objectsize(i32 addrspace(1)* %out) #0 {
  %alloca = alloca [17 x i32], align 4
  %alloca.bc = bitcast [17 x i32]* %alloca to i8*
  %size = call i32 @llvm.objectsize.i32.p0i8(i8* %alloca.bc, i1 false, i1 false, i1 false)
  store i32 %size, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @promote_alloca_used_twice_in_memcpy(
; CHECK: %i = bitcast double addrspace(3)* %arrayidx1 to i8 addrspace(3)*
; CHECK: %i1 = bitcast double addrspace(3)* %arrayidx2 to i8 addrspace(3)*
; CHECK: call void @llvm.memcpy.p3i8.p3i8.i64(i8 addrspace(3)* align 8 dereferenceable(16) %i, i8 addrspace(3)* align 8 dereferenceable(16) %i1, i64 16, i1 false)
define amdgpu_kernel void @promote_alloca_used_twice_in_memcpy(i32 %c) {
entry:
  %r = alloca double, align 8
  %arrayidx1 = getelementptr inbounds double, double* %r, i32 1
  %i = bitcast double* %arrayidx1 to i8*
  %arrayidx2 = getelementptr inbounds double, double* %r, i32 %c
  %i1 = bitcast double* %arrayidx2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 dereferenceable(16) %i, i8* align 8 dereferenceable(16) %i1, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: @promote_alloca_used_twice_in_memmove(
; CHECK: %i = bitcast double addrspace(3)* %arrayidx1 to i8 addrspace(3)*
; CHECK: %i1 = bitcast double addrspace(3)* %arrayidx2 to i8 addrspace(3)*
; CHECK: call void @llvm.memmove.p3i8.p3i8.i64(i8 addrspace(3)* align 8 dereferenceable(16) %i, i8 addrspace(3)* align 8 dereferenceable(16) %i1, i64 16, i1 false)
define amdgpu_kernel void @promote_alloca_used_twice_in_memmove(i32 %c) {
entry:
  %r = alloca double, align 8
  %arrayidx1 = getelementptr inbounds double, double* %r, i32 1
  %i = bitcast double* %arrayidx1 to i8*
  %arrayidx2 = getelementptr inbounds double, double* %r, i32 %c
  %i1 = bitcast double* %arrayidx2 to i8*
  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 dereferenceable(16) %i, i8* align 8 dereferenceable(16) %i1, i64 16, i1 false)
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="64,64" "amdgpu-waves-per-eu"="1,3" }
attributes #1 = { nounwind readnone }
