; RUN: not --crash llc -march=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: LLVM ERROR: Unsupported expression in static initializer: addrspacecast ([256 x i32] addrspace(3)* @lds.arr to [256 x i32] addrspace(4)*)

@lds.arr = unnamed_addr addrspace(3) global [256 x i32] undef, align 4

@gv_flatptr_from_lds = unnamed_addr addrspace(2) global i32 addrspace(4)* getelementptr ([256 x i32], [256 x i32] addrspace(4)* addrspacecast ([256 x i32] addrspace(3)* @lds.arr to [256 x i32] addrspace(4)*), i64 0, i64 8), align 4
