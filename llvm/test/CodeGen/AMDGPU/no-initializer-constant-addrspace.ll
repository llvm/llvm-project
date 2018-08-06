; RUN: llc -march=amdgcn -mcpu=tahiti -filetype=obj < %s | llvm-readobj -relocations -symbols | FileCheck %s -check-prefix=GCN
; RUN: llc -march=amdgcn -mcpu=tonga -filetype=obj < %s | llvm-readobj -relocations -symbols | FileCheck %s -check-prefix=GCN
; RUN: llc -march=r600 -mcpu=cypress -filetype=obj < %s | llvm-readobj -relocations -symbols | FileCheck %s -check-prefix=EG

; GCN: R_AMDGPU_REL32 extern_const_addrspace
; EG: R_AMDGPU_ABS32 extern_const_addrspace

; CHECK-DAG: Name: extern_const_addrspace
@extern_const_addrspace = external unnamed_addr addrspace(4) constant [5 x i32], align 4

; CHECK-DAG: Name: load_extern_const_init
define amdgpu_kernel void @load_extern_const_init(i32 addrspace(1)* %out) nounwind {
  %val = load i32, i32 addrspace(4)* getelementptr ([5 x i32], [5 x i32] addrspace(4)* @extern_const_addrspace, i64 0, i64 3), align 4
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-DAG: Name: undef_const_addrspace
@undef_const_addrspace = unnamed_addr addrspace(4) constant [5 x i32] undef, align 4

; CHECK-DAG: Name: undef_const_addrspace
define amdgpu_kernel void @load_undef_const_init(i32 addrspace(1)* %out) nounwind {
  %val = load i32, i32 addrspace(4)* getelementptr ([5 x i32], [5 x i32] addrspace(4)* @undef_const_addrspace, i64 0, i64 3), align 4
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}
