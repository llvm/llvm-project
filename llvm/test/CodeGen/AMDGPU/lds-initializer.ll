; RUN: not llc -march=amdgcn -mcpu=SI < %s 2>&1 | FileCheck %s
; RUN: not llc -march=amdgcn -mcpu=tonga < %s 2>&1 | FileCheck %s

; CHECK: error: unsupported initializer for address space in load_init_lds_global

@lds = addrspace(3) global [8 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8]

define void @load_init_lds_global(i32 addrspace(1)* %out, i1 %p) {
 %gep = getelementptr [8 x i32], [8 x i32] addrspace(3)* @lds, i32 0, i32 10
  %ld = load i32, i32 addrspace(3)* %gep
  store i32 %ld, i32 addrspace(1)* %out
  ret void
}
