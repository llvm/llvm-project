; RUN: not llc -march=amdgcn -mcpu=SI < %s 2>&1 | FileCheck %s
; RUN: not llc -march=amdgcn -mcpu=tonga < %s 2>&1 | FileCheck %s

; CHECK: error: unsupported initializer for address space in load_init_global_global

@lds = addrspace(1) global [256 x i32] zeroinitializer

define void @load_init_global_global(i32 addrspace(1)* %out, i1 %p) {
 %gep = getelementptr [256 x i32], [256 x i32] addrspace(1)* @lds, i32 0, i32 10
  %ld = load i32, i32 addrspace(1)* %gep
  store i32 %ld, i32 addrspace(1)* %out
  ret void
}
