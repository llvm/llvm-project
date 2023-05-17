; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -passes=amdgpu-promote-alloca < %s | FileCheck --enable-var-scope %s

declare void @llvm.memcpy.p5.p1.i32(ptr addrspace(5) nocapture, ptr addrspace(1) nocapture, i32, i1) #0
declare void @llvm.memcpy.p1.p5.i32(ptr addrspace(1) nocapture, ptr addrspace(5) nocapture, i32, i1) #0
declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) nocapture, ptr addrspace(5) nocapture, i64, i1) #0

declare void @llvm.memmove.p5.p1.i32(ptr addrspace(5) nocapture, ptr addrspace(1) nocapture, i32, i1) #0
declare void @llvm.memmove.p1.p5.i32(ptr addrspace(1) nocapture, ptr addrspace(5) nocapture, i32, i1) #0
declare void @llvm.memmove.p5.p5.i64(ptr addrspace(5) nocapture, ptr addrspace(5) nocapture, i64, i1) #0

declare void @llvm.memset.p5.i32(ptr addrspace(5) nocapture, i8, i32, i1) #0

declare i32 @llvm.objectsize.i32.p5(ptr addrspace(5), i1, i1, i1) #1

; CHECK-LABEL: @promote_with_memcpy(
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds [64 x [17 x i32]], ptr addrspace(3) @promote_with_memcpy.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memcpy.p3.p1.i32(ptr addrspace(3) align 4 [[GEP]], ptr addrspace(1) align 4 %in, i32 68, i1 false)
; CHECK: call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 [[GEP]], i32 68, i1 false)
define amdgpu_kernel void @promote_with_memcpy(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %alloca = alloca [17 x i32], align 4, addrspace(5)
  call void @llvm.memcpy.p5.p1.i32(ptr addrspace(5) align 4 %alloca, ptr addrspace(1) align 4 %in, i32 68, i1 false)
  call void @llvm.memcpy.p1.p5.i32(ptr addrspace(1) align 4 %out, ptr addrspace(5) align 4 %alloca, i32 68, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_memmove(
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds [64 x [17 x i32]], ptr addrspace(3) @promote_with_memmove.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memmove.p3.p1.i32(ptr addrspace(3) align 4 [[GEP]], ptr addrspace(1) align 4 %in, i32 68, i1 false)
; CHECK: call void @llvm.memmove.p1.p3.i32(ptr addrspace(1) align 4 %out, ptr addrspace(3) align 4 [[GEP]], i32 68, i1 false)
define amdgpu_kernel void @promote_with_memmove(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %alloca = alloca [17 x i32], align 4, addrspace(5)
  call void @llvm.memmove.p5.p1.i32(ptr addrspace(5) align 4 %alloca, ptr addrspace(1) align 4 %in, i32 68, i1 false)
  call void @llvm.memmove.p1.p5.i32(ptr addrspace(1) align 4 %out, ptr addrspace(5) align 4 %alloca, i32 68, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_memset(
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds [64 x [17 x i32]], ptr addrspace(3) @promote_with_memset.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call void @llvm.memset.p3.i32(ptr addrspace(3) align 4 [[GEP]], i8 7, i32 68, i1 false)
define amdgpu_kernel void @promote_with_memset(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %alloca = alloca [17 x i32], align 4, addrspace(5)
  call void @llvm.memset.p5.i32(ptr addrspace(5) align 4 %alloca, i8 7, i32 68, i1 false)
  ret void
}

; CHECK-LABEL: @promote_with_objectsize(
; CHECK: [[PTR:%[0-9]+]] = getelementptr inbounds [64 x [17 x i32]], ptr addrspace(3) @promote_with_objectsize.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: call i32 @llvm.objectsize.i32.p3(ptr addrspace(3) [[PTR]], i1 false, i1 false, i1 false)
define amdgpu_kernel void @promote_with_objectsize(ptr addrspace(1) %out) #0 {
  %alloca = alloca [17 x i32], align 4, addrspace(5)
  %size = call i32 @llvm.objectsize.i32.p5(ptr addrspace(5) %alloca, i1 false, i1 false, i1 false)
  store i32 %size, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @promote_alloca_used_twice_in_memcpy(
; CHECK: call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) align 8 dereferenceable(16) %arrayidx1, ptr addrspace(3) align 8 dereferenceable(16) %arrayidx2, i64 16, i1 false)
define amdgpu_kernel void @promote_alloca_used_twice_in_memcpy(i32 %c) {
entry:
  %r = alloca double, align 8, addrspace(5)
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(5) %r, i32 1
  %arrayidx2 = getelementptr inbounds double, ptr addrspace(5) %r, i32 %c
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 8 dereferenceable(16) %arrayidx1, ptr addrspace(5) align 8 dereferenceable(16) %arrayidx2, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: @promote_alloca_used_twice_in_memmove(
; CHECK: call void @llvm.memmove.p3.p3.i64(ptr addrspace(3) align 8 dereferenceable(16) %arrayidx1, ptr addrspace(3) align 8 dereferenceable(16) %arrayidx2, i64 16, i1 false)
define amdgpu_kernel void @promote_alloca_used_twice_in_memmove(i32 %c) {
entry:
  %r = alloca double, align 8, addrspace(5)
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(5) %r, i32 1
  %arrayidx2 = getelementptr inbounds double, ptr addrspace(5) %r, i32 %c
  call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) align 8 dereferenceable(16) %arrayidx1, ptr addrspace(5) align 8 dereferenceable(16) %arrayidx2, i64 16, i1 false)
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="64,64" "amdgpu-waves-per-eu"="1,3" }
attributes #1 = { nounwind readnone }
