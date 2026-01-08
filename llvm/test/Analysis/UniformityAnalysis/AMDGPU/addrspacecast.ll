; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s --check-prefix=UNI
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s --check-prefix=DIV

; UNI: ALL VALUES UNIFORM
; DIV: DIVERGENT:   %cast = addrspacecast ptr addrspace(5) %alloca to ptr
; DIV: DIVERGENT:   %cast.1 = call ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p5(ptr addrspace(5) %alloca)
define void @foo() {
  %alloca = alloca i32, align 4, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %alloca to ptr
  store i32 1, ptr %cast
  %cast.1 = call ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p5(ptr addrspace(5) %alloca)
  store i32 2, ptr %cast.1
  ret void
}
