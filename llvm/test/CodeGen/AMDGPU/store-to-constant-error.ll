; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o /dev/null %s 2>&1 | FileCheck -check-prefix=SDAG %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o /dev/null %s 2>&1 | FileCheck -check-prefix=GISEL %s

; SDAG: LLVM ERROR: Cannot select: {{[a-z0-9]+}}: ch = store<(store (s32) into %ir.ptr.load, addrspace 4)>
; GISEL: LLVM ERROR: cannot select: G_STORE %{{[0-9]+}}:vgpr(s32), %{{[0-9]+}}:vgpr(p4) :: (store (s32) into %ir.ptr.load, addrspace 4) (in function: store_to_constant_i32)
define amdgpu_kernel void @store_to_constant_i32(ptr addrspace(4) %ptr) {
bb:
  store i32 1, ptr addrspace(4) %ptr, align 4
  ret void
}
