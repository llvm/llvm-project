; RUN: not --crash llc -mtriple=amdgcn -verify-machineinstrs -amdgpu-enable-lower-module-lds=false < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: LLVM ERROR: Unsupported expression in static initializer: addrspacecast (ptr addrspace(3) @lds.arr to ptr addrspace(4))

@lds.arr = unnamed_addr addrspace(3) global [256 x i32] undef, align 4

@gv_flatptr_from_lds = unnamed_addr addrspace(2) global ptr addrspace(4) getelementptr ([256 x i32], ptr addrspace(4) addrspacecast (ptr addrspace(3) @lds.arr to ptr addrspace(4)), i64 0, i64 8), align 4
