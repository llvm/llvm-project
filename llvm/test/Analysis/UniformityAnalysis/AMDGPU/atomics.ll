; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %orig = atomicrmw xchg ptr %ptr, i32 %val seq_cst
define amdgpu_kernel void @test1(ptr %ptr, i32 %val) #0 {
  %orig = atomicrmw xchg ptr %ptr, i32 %val seq_cst
  store i32 %orig, ptr %ptr
  ret void
}

; CHECK: DIVERGENT: %orig = cmpxchg ptr %ptr, i32 %cmp, i32 %new seq_cst seq_cst
define amdgpu_kernel void @test2(ptr %ptr, i32 %cmp, i32 %new) {
  %orig = cmpxchg ptr %ptr, i32 %cmp, i32 %new seq_cst seq_cst
  %val = extractvalue { i32, i1 } %orig, 0
  store i32 %val, ptr %ptr
  ret void
}

; CHECK: DIVERGENT: %ret = call i32 @llvm.amdgcn.global.atomic.csub.p1(ptr addrspace(1) %ptr, i32 %val)
define amdgpu_kernel void @test_atomic_csub_i32(ptr addrspace(1) %ptr, i32 %val) #0 {
  %ret = call i32 @llvm.amdgcn.global.atomic.csub.p1(ptr addrspace(1) %ptr, i32 %val)
  store i32 %ret, i32 addrspace(1)* %ptr, align 4
  ret void
}

; CHECK: DIVERGENT: %val = call i32 @llvm.amdgcn.atomic.cond.sub.u32.p3(ptr addrspace(3) %gep, i32 %in)
define amdgpu_kernel void @test_ds_atomic_cond_sub_rtn_u32(ptr addrspace(3) %addr, i32 %in, ptr addrspace(3) %use) #0 {
entry:
  %gep = getelementptr i32, ptr addrspace(3) %addr, i32 4
  %val = call i32 @llvm.amdgcn.atomic.cond.sub.u32.p3(ptr addrspace(3) %gep, i32 %in)
  store i32 %val, ptr addrspace(3) %use
  ret void
}

; CHECK: DIVERGENT: %val = call i32 @llvm.amdgcn.atomic.cond.sub.u32.p0(ptr %gep, i32 %in)
define amdgpu_kernel void @test_flat_atomic_cond_sub_u32(ptr %addr, i32 %in, ptr %use) #0 {
entry:
  %gep = getelementptr i32, ptr %addr, i32 4
  %val = call i32 @llvm.amdgcn.atomic.cond.sub.u32.p0(ptr %gep, i32 %in)
  store i32 %val, ptr %use
  ret void
}

; CHECK: DIVERGENT: %val = call i32 @llvm.amdgcn.atomic.cond.sub.u32.p1(ptr addrspace(1) %gep, i32 %in)
define amdgpu_kernel void @test_global_atomic_cond_u32(ptr addrspace(1) %addr, i32 %in, ptr addrspace(1) %use) #0 {
entry:
  %gep = getelementptr i32, ptr addrspace(1) %addr, i32 4
  %val = call i32 @llvm.amdgcn.atomic.cond.sub.u32.p1(ptr addrspace(1) %gep, i32 %in)
  store i32 %val, ptr addrspace(1) %use
  ret void
}

; CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.raw.buffer.atomic.cond.sub.u32.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
define float @test_raw_buffer_atomic_cond_sub_u32(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
entry:
  %orig = call i32 @llvm.amdgcn.raw.buffer.atomic.cond.sub.u32.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

; CHECK: DIVERGENT: %orig = call i32 @llvm.amdgcn.struct.buffer.atomic.cond.sub.u32.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i32 0, i32 0)
define float @test_struct_buffer_atomic_cond_sub_u32(<4 x i32> inreg %rsrc, i32 inreg %data) #0 {
entry:
  %orig = call i32 @llvm.amdgcn.struct.buffer.atomic.cond.sub.u32.i32(i32 %data, <4 x i32> %rsrc, i32 0, i32 0, i32 0, i32 0)
  %r = bitcast i32 %orig to float
  ret float %r
}

declare i32 @llvm.amdgcn.global.atomic.csub.p1(ptr addrspace(1) nocapture, i32) #1
declare i32 @llvm.amdgcn.atomic.cond.sub.u32.p3(ptr addrspace(3), i32) #1
declare i32 @llvm.amdgcn.atomic.cond.sub.u32.p0(ptr, i32) #1
declare i32 @llvm.amdgcn.atomic.cond.sub.u32.p1(ptr addrspace(1), i32) #1
declare i32 @llvm.amdgcn.raw.buffer.atomic.cond.sub.u32.i32(i32, <4 x i32>, i32, i32, i32) #1
declare i32 @llvm.amdgcn.struct.buffer.atomic.cond.sub.u32.i32(i32, <4 x i32>, i32, i32, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind willreturn }
