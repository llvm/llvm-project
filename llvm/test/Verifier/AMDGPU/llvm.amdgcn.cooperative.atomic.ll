; RUN: not opt -mtriple=amdgcn -mcpu=gfx1250 -passes=verify  -disable-output <%s 2>&1 | FileCheck %s

; CHECK: cooperative atomic intrinsics require a generic or global pointer
define i32 @load_local_as(ptr addrspace(3) noundef %addr)  {
entry:
  %res = tail call i32 @llvm.amdgcn.cooperative.atomic.load.32x4B.p3(ptr addrspace(3) %addr, i32 0, metadata !0)
  ret i32 %res
}

; CHECK: cooperative atomic intrinsics require a generic or global pointer
define i32 @load_private_as(ptr addrspace(5) noundef %addr)  {
entry:
  %res = tail call i32 @llvm.amdgcn.cooperative.atomic.load.32x4B.p5(ptr addrspace(5) %addr, i32 0, metadata !0)
  ret i32 %res
}

; CHECK: cooperative atomic intrinsics require a generic or global pointer
define void @store_local_as(ptr addrspace(3) noundef %addr, i32 noundef %val)  {
entry:
  tail call void @llvm.amdgcn.cooperative.atomic.store.32x4B.p3(ptr addrspace(3) %addr, i32 %val, i32 0, metadata !0)
  ret void
}

; CHECK: cooperative atomic intrinsics require a generic or global pointer
define void @store_private_as(ptr addrspace(5) noundef %addr, i32 noundef %val)  {
entry:
  tail call void @llvm.amdgcn.cooperative.atomic.store.32x4B.p5(ptr addrspace(5) %addr, i32 %val, i32 0, metadata !0)
  ret void
}


; CHECK: cooperative atomic intrinsics require that the last argument is a metadata string
define i32 @test_empty_md(ptr noundef readonly %addr)  {
entry:
  %0 = tail call i32 @llvm.amdgcn.cooperative.atomic.load.32x4B.p0(ptr %addr, i32 1, metadata !{})
  ret i32 %0
}

; CHECK: cooperative atomic intrinsics require that the last argument is a metadata string
define i32 @test_no_md_str(ptr noundef readonly %addr)  {
entry:
  %0 = tail call i32 @llvm.amdgcn.cooperative.atomic.load.32x4B.p0(ptr %addr, i32 1, metadata !{!{}})
  ret i32 %0
}

!0 = !{ !"" }
