; RUN: not llc -march=amdgcn -mcpu=tahiti < %s 2>&1 | FileCheck %s
; RUN: not llc -march=amdgcn -mcpu=tonga < %s 2>&1 | FileCheck %s

; CHECK: in function load_zeroinit_lds_global{{.*}}: unsupported initializer for address space

@lds = addrspace(3) global [256 x i32] zeroinitializer

define amdgpu_kernel void @load_zeroinit_lds_global(i32 addrspace(1)* %out, i1 %p) {
 %gep = getelementptr [256 x i32], [256 x i32] addrspace(3)* @lds, i32 0, i32 10
  %ld = load i32, i32 addrspace(3)* %gep
  store i32 %ld, i32 addrspace(1)* %out
  ret void
}
