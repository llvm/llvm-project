; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s

define protected amdgpu_kernel void @foo(ptr addrspace(1) %arg, ptr addrspace(1) %arg1) {
bb:
  %tmp = addrspacecast ptr addrspace(5) null to ptr
  %tmp2 = call i64 @eggs(ptr undef) #1
  %tmp3 = load ptr, ptr %tmp, align 8
  %tmp4 = getelementptr inbounds i64, ptr %tmp3, i64 undef
  store i64 %tmp2, ptr %tmp4, align 8
  ret void
}

declare hidden i64 @eggs(ptr)
