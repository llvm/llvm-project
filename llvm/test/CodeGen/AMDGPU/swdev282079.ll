; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s

define protected amdgpu_kernel void @foo(i64 addrspace(1)* %arg, i64 addrspace(1)* %arg1) {
bb:
  %tmp = addrspacecast i64* addrspace(5)* null to i64**
  %tmp2 = call i64 @eggs(i64* undef) #1
  %tmp3 = load i64*, i64** %tmp, align 8
  %tmp4 = getelementptr inbounds i64, i64* %tmp3, i64 undef
  store i64 %tmp2, i64* %tmp4, align 8
  ret void
}

declare hidden i64 @eggs(i64*)
