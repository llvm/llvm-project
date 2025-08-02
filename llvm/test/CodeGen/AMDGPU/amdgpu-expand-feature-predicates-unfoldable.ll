; REQUIRES: amdgpu-registered-target

; RUN: not opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -passes='amdgpu-expand-feature-predicates' < %s 2>&1 | FileCheck %s

; CHECK: error:{{.*}}in function kernel void (ptr addrspace(1), i32, ptr addrspace(1)): Impossible to constant fold feature predicate: @llvm.amdgcn.is.gfx803 = private addrspace(1) constant i1 false used by   %call = call i1 %1(i1 zeroext false), please simplify.

@llvm.amdgcn.is.gfx803 = external addrspace(1) externally_initialized constant i1

declare void @llvm.amdgcn.s.sleep(i32 immarg) #1

define amdgpu_kernel void @kernel(ptr addrspace(1) readnone captures(none) %p.coerce, i32 %x, ptr addrspace(1) %pfn.coerce) {
entry:
  %0 = ptrtoint ptr addrspace(1) %pfn.coerce to i64
  %1 = inttoptr i64 %0 to ptr
  %2 = ptrtoint ptr addrspace(1) %pfn.coerce to i64
  %3 = load i1, ptr addrspace(1) @llvm.amdgcn.is.gfx803, align 1
  %call = call i1 %1(i1 zeroext %3)
  br i1 %call, label %if.gfx803, label %if.end

if.gfx803:
  call void @llvm.amdgcn.s.sleep(i32 0)
  br label %if.end

if.end:
  ret void
}

attributes #1 = { nocallback nofree nosync nounwind willreturn }
